#!/usr/bin/env python3

# basic libraries

# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve
from keras import regularizers
from tensorflow.keras.optimizers import SGD
#dataset
df = pd.read_excel("Conversion dataset Sep 8 to send.xlsx")
df = df.dropna()
#print(df)
x = df.drop('Y',axis=1)
y = df['Y']
# train-test split
trainX, testX, trainy, testy = train_test_split(x, y, test_size=0.3, random_state=13)

# define the neural network model
def define_model(n_input):
	# define model
	model = Sequential()
	# define first hidden layer and visible layer
	model.add(Dense(10, input_dim=n_input, activation='relu', kernel_initializer='he_uniform'))
	# define output layer
	model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2	(0.001)))
	# define loss and optimizer
	opt = SGD(lr=0.01)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model


# define the model
n_input = trainX.shape[1]
model = define_model(n_input)
# fit model
weights = {0:0.45, 1:5.5}
history = model.fit(trainX, trainy, class_weight=weights, epochs=30	, verbose=0)
# make predictions on the test dataset
yhat = model.predict(testX)
# evaluate the ROC AUC of the predictions
score = roc_auc_score(testy, yhat)
print('ROC AUC: %.3f' % score)


# evaluate the keras model
accuracy = model.evaluate(testX, yhat)
print('Accuracy: %.2f' % (accuracy*100))


# predict probabilities for test set
yhat_probs = model.predict(testX, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(testX, verbose=0)

# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]


# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(testy, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(testy, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(testy, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(testy, yhat_classes)
print('F1 score: %f' % f1)
# confusion matrix
matrix = confusion_matrix(testy, yhat_classes)
print(matrix)
