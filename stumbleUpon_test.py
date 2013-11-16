import pandas as pd
import numpy as np
from ensemble.EnsembleUtils import auc_error
from sklearn import cross_validation
from sklearn.preprocessing import Imputer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import words
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from ensemble.EnsembleSelection import EnsembleSelection
from ensemble.EnsembleClassifier import EnsembleClassifier
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.metrics import auc

import json

if __name__ == "__main__":
	links = pd.read_csv('./datasets/stumbleupon_train.tsv',sep='\t')
	
	colnames = links.columns.tolist()
	json_poilerplate = links.icol(2).tolist()
	n = len(json_poilerplate)
	y_label = links['label'] 
	text = []
	for i in range(0,n):
	    try:
	        obj =  json.loads(json_poilerplate[i])
	        text.append(obj['title'])
	    except KeyError:
	        text.append(obj['body'])
	        print "No description"
	normalized_text = []
	for i in text:
	    if i is not None:
	        l = wordpunct_tokenize(i)
	        w = [a for a in l if not a in stopwords.words('english') if not len(a)==1]
	        normalized_text.append(' '.join(w).encode('utf-8').strip())
	    else :   
	        normalized_text.append(' ')
	X_TRAIN , X_TEST , Y_TRAIN , Y_TEST = cross_validation.train_test_split(normalized_text,links['label'].tolist(), test_size=0.1, random_state=0)
	vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.7,stop_words='english',lowercase=True,use_idf=True)
	X_TRAIN = vectorizer.fit_transform(X_TRAIN)
	X_TEST = vectorizer.transform(X_TEST)
	selection = EnsembleSelection()
	clf = EnsembleClassifier()
	models = []
	models = (selection.generate_logistic_regression_classifiers()+selection.generate_bernoulli_nb_classifiers(300)+selection.generate_multionomial_nb_classifiers(300))
	ensemble =  selection.form_ensemble(X_TRAIN,Y_TRAIN,models,30,error_metric_name='auc',replacement=False)
	y_predicted = clf.predict(X_TEST,ensemble)
	print auc_error(y_predicted,Y_TEST)






