#!/usr/bin/env python3

# basic libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_classification
#import keras
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasClassifier

#import model and matrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve
from sklearn.metrics import f1_score

#dataset
df = pd.read_excel("Conversion dataset Sep 8 to send.xlsx")
df = df.dropna()
#print(df)

# check the distribution
distribution = df["Y"].value_counts()/df.shape[0]
print(distribution)

# scatter plot
plt.figure(figsize=(10,5))
#im = sns.scatterplot(data=df,x="age",y="BMI",hue="Y")
#plt.show()

#LogisticRegression
# split dataset into x,y
x = df.drop('Y',axis=1)
y = df['Y']
# train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=13)

# # define model
# lg1 = LogisticRegression(random_state=13, class_weight=None, max_iter = 1000)
# # fit it
# lg1.fit(X_train,y_train)
# # test
# y_pred = lg1.predict_proba(X_test)[:,1]
# y_pred = np.where(y_pred>0.45, 1, 0)


# # performance
# print(f'Accuracy Score: {accuracy_score(y_test,y_pred)}')
# print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
# print(f'Area Under Curve: {roc_auc_score(y_test, y_pred)}')
# print(f'Recall score: {recall_score(y_test,y_pred)}')

# # define class weights
# w = {0:1, 1:100}
# # define model
# lg2 = LogisticRegression(random_state=13, class_weight=w, max_iter = 1000)
# # fit it
# lg2.fit(X_train,y_train)
# # test
# y_pred = lg2.predict_proba(X_test)[:,1]
# y_pred = np.where(y_pred>0.5, 1, 0)
# # performance
# print(f'Accuracy Score: {accuracy_score(y_test,y_pred)}')
# print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
# print(f'Area Under Curve: {roc_auc_score(y_test, y_pred)}')
# print(f'Recall score: {recall_score(y_test,y_pred)}')


# # define weight hyperparameter
# w = [{0:1000,1:100},{0:1000,1:10}, {0:1000,1:1.0},
     # {0:500,1:1.0}, {0:400,1:1.0}, {0:300,1:1.0}, {0:200,1:1.0},
     # {0:150,1:1.0}, {0:100,1:1.0}, {0:99,1:1.0}, {0:10,1:1.0},
     # {0:0.01,1:1.0}, {0:0.01,1:10}, {0:0.01,1:100},
     # {0:0.001,1:1.0}, {0:0.005,1:1.0}, {0:1.0,1:1.0},
     # {0:1.0,1:0.1}, {0:10,1:0.1}, {0:100,1:0.1},
     # {0:10,1:0.01}, {0:1.0,1:0.01}, {0:1.0,1:0.001}, {0:1.0,1:0.005},
     # {0:1.0,1:10}, {0:1.0,1:99}, {0:1.0,1:100}, {0:1.0,1:150},
     # {0:1.0,1:200}, {0:1.0,1:300},{0:1.0,1:400},{0:1.0,1:500},
     # {0:1.0,1:1000}, {0:10,1:1000},{0:100,1:1000} ]
# hyperparam_grid = {"class_weight": w }

# # define model
# lg3 = LogisticRegression(random_state=13, max_iter=1000)
# # define evaluation procedure
# grid = GridSearchCV(lg3,hyperparam_grid,scoring="roc_auc", cv=10, n_jobs=-1, refit=True)
# grid.fit(x,y)
# print(f'Best score: {grid.best_score_} with param: {grid.best_params_}')




# weighted logistic regression for class imbalance with heuristic weights
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
# generate dataset

# define model
model = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1000)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))

#F1 score
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print(model.score(X_test, y_test))
f1score = f1_score(y_test, predictions)
print('F1 score:', f1score)

# AUC plot; ref: https://www.kaggle.com/code/kanncaa1/roc-curve-with-k-fold-cv/notebook
from sklearn.metrics import roc_curve, auc
from scipy import interp
fig1 = plt.figure(figsize=[12,12])
tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)
i = 1
for train, test in cv.split(x,y):
    prediction = model.fit(x.iloc[train],y.iloc[train]).predict_proba(x.iloc[test])
    fpr, tpr, t = roc_curve(y.iloc[test], prediction[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.02f)' % (i, roc_auc))
    i= i+1

mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.02f )' % (mean_auc),lw=2, alpha=1)
plt.plot([0, 1], [0, 1], color = 'black', linewidth = 2)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right", prop = {"size":5})
plt.savefig("lg_auc.jpg", dpi=1000)
plt.show()
