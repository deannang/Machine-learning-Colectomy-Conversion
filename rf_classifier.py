
# first neural network with keras tutorial
from numpy import loadtxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve
from sklearn.preprocessing import LabelEncoder#for train test splitting
from sklearn.model_selection import train_test_split#for decision tree object
from sklearn.tree import DecisionTreeClassifier#for checking testing results
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix#for visualizing tree
from sklearn.tree import plot_tree
df = pd.read_excel("Conversion dataset Sep 8 to send.xlsx")
df = df.dropna()
X = df.drop('Y',axis=1)
y = df['Y']
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.3, random_state = 42)
print("Training split input- ", X_train.shape)
print("Testing split input- ", X_test.shape)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
print (scaler.data_max_)
X_train_trans = scaler.transform(X_train)
X_test_trans  = scaler.transform(X_test)
y_train_vals  = y_train.values
y_test_vals   = y_test.values

model   = RandomForestClassifier()
param_grid = {
    'n_estimators':[5, 20, 50],
    'criterion' : ["gini","entropy"],
    "max_depth" : [2, 10, 20],
    'min_samples_split' : [2, 10, 50],
    'max_features': ['auto', 'sqrt', 'log2'],
    'class_weight': ["balanced"],
    'n_jobs' : [6]
    #[{0:100,1:1}, {0:10,1:1}, {0:1,1:1}, {0:1,1:10}, {0:1,1:100}]
}

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv,
                    scoring='roc_auc')
grid_result = grid.fit(X_train_trans, y_train_vals)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds  = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# AUC curve
from sklearn.metrics import plot_roc_curve
y_pred = grid_result.predict(X_test_trans)
plot = plot_roc_curve(grid, X_test_trans, y_test_vals)
plt.plot([0, 1], [0, 1], color = 'black', linewidth = 2)
plt.savefig("rf_clf_auc.jpg",dpi=1000)
plt.show()
