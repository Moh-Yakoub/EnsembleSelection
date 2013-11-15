from sklearn.metrics import auc
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
import random
import logging
class EnsembleSelection:
	'''
	This class implements the EnsembleSelection mechanism
	'''
	'''
	All of the metrics used for classification model selection
	'''
	classification_error_metrics = {
	'auc':auc,
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

	def generate_logistic_regression_classifiers(self,n=121):
		self.logger.info('Generating a set of %d logistic regression classifiers',n)

		models = []
		for i in range(0,n):
			penalty_var = 'l1' if random.random() > 0.5 else 'l2'
			tol_var = random.random()
			C_var =random.random()
			fit_intercept_var = True if random.random() > 0.5 else False
			models.append(LogisticRegression(penalty=penalty_var,C=C_var,fit_intercept = fit_intercept_var,tol = tol_var))

		self.logger.info('Generating a set of %d logistic regression classifiers was created successfully!',n)
		return models

	def generate_multionomial_nb_classifiers(self,n=121):
		self.logger.info('Generating a set of %d Multinomial naive bayes classifiers',n)

		models = []
		for i in range(0,n):
			alpha_var = random.random()
			fit_prior_var =  True if random.random() > 0.5 else False
			models.append(MultinomialNB(alpha=alpha_var,fit_prior=fit_prior_var))

		self.logger.info('Generating a set of %d Multinomial naive bayes classifiers was created successfully!',n)
		return models

	def generate_bernoulli_nb_classifiers(self,n=121):
		self.logger.info('Generating a set of %d Bernoulli naive bayes classifiers',n)

		models = []
		for i in range(0,n):
			alpha_var = random.random()
			fit_prior_var =  True if random.random() > 0.5 else False
			binarize_var =  random.random()
			models.append(BernoulliNB(alpha=alpha_var,fit_prior=fit_prior_var,binarize = binarize_var))

		self.logger.info('Generating a set of %d Bernoulli naive bayes classifiers was created successfully!',n)
		return models


		

if __name__ == "__main__":
	selection = EnsembleSelection()
	selection.generate_logistic_regression_classifiers()
	selection.generate_bernoulli_nb_classifiers()
	selection.generate_multionomial_nb_classifiers()







