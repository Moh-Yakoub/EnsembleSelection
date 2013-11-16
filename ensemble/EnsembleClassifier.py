from operator import add
import logging
from EnsembleUtils import vote
class EnsembleClassifier:
	def __init__(self):
		FORMAT = '%(asctime)s  %(message)s'
		logging.basicConfig(format=FORMAT)
		self.logger = logging.getLogger('ensemble-selection-logger')
		self.logger.setLevel(logging.INFO)

	def predict(self,X,ensemble):
		self.logger.info("Classifying using an ensemble of count %d models",len(ensemble))
		model = ensemble[0]
		predictions_list = [model.predict(X).tolist()]
		for i in range(1,len(ensemble)):
			self.logger.info("Classifying using ensemble model %d ",i)
			predicted = ensemble[i].predict(X)
			predictions_list.append(ensemble[i].predict(X).tolist())
		self.logger.info("Aggregating using voting")
		return vote(predictions_list)




