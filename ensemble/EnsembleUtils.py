from collections import Counter
from sklearn import metrics
'''
This module defines a set of useful utility models used for ensemble selection
'''

'''
This method is used for aggregating a set of classified lists using the voting method

params:
l : a list of lists containing a list of predicted classifications
'''

def vote(l):
	n = len(l[0])
	result = []
	for i in range(0,n):
		ithlist = [x[i] for x in l]
		result.append(Counter(ithlist).most_common(1)[0][0])

	return result

def auc_error(pred,Y_TEST):
	fpr, tpr, thresholds = metrics.roc_curve(Y_TEST, pred, pos_label=1)
	auc = metrics.auc(fpr,tpr)
	return auc  
