# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 17:17:05 2019

@author: Alimo
"""


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

# Load and split the data
dataset_trn = pd.read_csv('bb_2000s_train.csv')
dataset_tst = pd.read_csv('bb_2000s_test.csv')

X = dataset_trn.iloc[:, 3:15].values
y = dataset_trn.iloc[:, 15].values
XT = dataset_tst.iloc[:, 0:12].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construct some pipelines


pipe_svm = Pipeline([('scl', StandardScaler()),
			('clf', svm.SVC(random_state=42))])

pipe_svm_pca = Pipeline([('scl', StandardScaler()),
			('pca', PCA(n_components=2)),
			('clf', svm.SVC(random_state=42))])

pipe_KNN = Pipeline([('scl', StandardScaler()), ('clf', KNeighborsClassifier())])

pipe_MNB = Pipeline([('clf', MultinomialNB())])
			
# Set grid search params
C = [1,2, 5, 10]


grid_params_svm = [{'clf__kernel': ['linear', 'rbf'], 
		'clf__C': C}]

grid_params_KNN = [{'clf__n_neighbors' : [3,5,8]}]

grid_params_MNB = [{'clf__alpha' : [1.0]}]

# Construct grid searches
jobs = -1

gs_svm = GridSearchCV(estimator=pipe_svm,
			param_grid=grid_params_svm,
			scoring='accuracy',
			cv=3,
			n_jobs=jobs)

gs_svm_pca = GridSearchCV(estimator=pipe_svm_pca,
			param_grid=grid_params_svm,
			scoring='accuracy',
			cv=3,
			n_jobs=jobs)

gs_KNN = GridSearchCV(estimator=pipe_KNN,
			param_grid=grid_params_KNN,
			scoring='accuracy',
			cv=3,
			n_jobs=jobs)

gs_MNB = GridSearchCV(estimator=pipe_MNB,
			param_grid=grid_params_MNB,
			scoring='accuracy',
			cv=3,
			n_jobs=jobs)

# List of pipelines for ease of iteration
grids = [gs_svm, gs_KNN, gs_MNB]

# Dictionary of pipelines and classifier types for ease of reference
grid_dict = {0: 'Support Vector Machine', 1: 'KNN', 2: 'MNB'}

# Fit the grid search objects
if __name__ == "__main__":
    print('Performing model optimizations...')
    best_acc = 0.0
    best_clf = 0
    best_gs = ''
    for idx, gs in enumerate(grids):
    	print('\nEstimator: %s' % grid_dict[idx])	
    	# Fit grid search	
    	gs.fit(X_train, y_train)
    	# Best params
    	print('Best params: %s' % gs.best_params_)
    	# Best training data accuracy
    	print('Best training accuracy: %.3f' % gs.best_score_)
    	# Predict on test data with best params
    	y_pred = gs.predict(X_test)
    	# Test data accuracy of model with best params
    	print('Test set accuracy score for best params: %.3f ' % accuracy_score(y_test, y_pred))
    	# Track best (highest test accuracy) model
    	if accuracy_score(y_test, y_pred) > best_acc:
            best_acc = accuracy_score(y_test, y_pred)
            best_gs = gs
            best_clf = idx
            output = gs.predict(XT)
            with open('predictions.txt', 'w') as outfile:
                for items in output:
                    outfile.write(str(items) + '\n')
                outfile.close()
    print('\nClassifier with best test set accuracy: %s' % grid_dict[best_clf])
    
    # Save best grid search pipeline to file
    dump_file = 'best_gs_pipeline.pkl'
    joblib.dump(best_gs, dump_file, compress=1)
    print('\nSaved %s grid search pipeline to file: %s' % (grid_dict[best_clf], dump_file))