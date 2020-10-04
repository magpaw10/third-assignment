import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

# built-in functions needed for task 2
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score

import seaborn as sns
import matplotlib.pyplot as plt

data_file = os.path.join('data', 'train.tsv')
test_data_file = os.path.join('data', 'test.tsv')
results_file = os.path.join('data', 'results.tsv')
output_file = os.path.join('data', 'out.tsv')

df_names = ['Occupancy', 'Date', 'Temperature', 'Humidity',
            'Light', 'CO2', 'HumidityRatio']

df = pd.read_csv(data_file, sep='\t', names=df_names)
df = df.dropna()

print (df)

occupancy_percentage = sum(df.Occupancy) / len(df)
print("Occupancy percentage is: " + str(occupancy_percentage))
print("Zero rule model accuracy on training set is: "
      + str(1 - occupancy_percentage))

# logistic regression classifier on one independent variable => CO2
clf = LogisticRegression()
X_train_co2 = df[['CO2']]
y_train_co2 = df.Occupancy

clf.fit(X_train_co2, y_train_co2)
y_train_pred_co2 = clf.predict(X_train_co2)

clf_accuracy_co2 = sum(y_train_co2 == y_train_pred_co2) / len(df)
print("Training set accuracy for logisitic regression model "
      + "on CO2 variable:\n" + str(clf_accuracy_co2))

# logistic regression classifier on one independent variable => Temperature
clf = LogisticRegression()
X_train = df[['Temperature']]
y_train = df.Occupancy

clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)

clf_accuracy = sum(y_train == y_train_pred) / len(df)
print ("Training set accuracy for logisitic regression model "
      + "on Temperature variable:\n" + str(clf_accuracy))

# zero rule model accuracy on training set
print (accuracy_score(y_train, np.zeros(len(y_train))))
# accurancy => built-in function use
print (accuracy_score(y_train, y_train_pred))

# accuracy, sensitivity and specificity built upon conf_matrix for one variable
conf_matrix = confusion_matrix(y_train, y_train_pred)
tn, fp, fn, tp = conf_matrix.ravel()
print ("True Positive: " + str(tp))
print ("False posiotive: "+ str(fp))
print ("False negative: " +str(fn))
print ("True negative: " + str(tn))

accuracy = (tn + tp) / (tp + fp + fn + tn)
print ("Accuracy: " + str(accuracy))
sensitivity = tp / (tp + fn)
print ("Sensitivity: " + str(sensitivity))
specificity = tn / (tn + fp)
print ("Specificity: " + str(specificity))

# logistic regression classifier on all but date independent variables
clf_all = LogisticRegression()
X_train_all = df[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']]
clf_all.fit(X_train_all, y_train)

y_train_pred_all = clf_all.predict(X_train_all)

clf_all_accuracy = accuracy_score(y_train, y_train_pred_all)
print("Training set accuracy for logisitic regression model " +
      "on all but date variable: " + str(clf_all_accuracy))

conf_matrix = confusion_matrix(y_train, y_train_pred_all)
tn, fp, fn, tp = conf_matrix.ravel()
print ("True Positive: " + str(tp))
print ("False posiotive: "+ str(fp))
print ("False negative: " +str(fn))
print ("True negative: " + str(tn))

accuracy = (tn + tp) / (tp + fp + fn + tn)
print ("Accuracy: " + str(accuracy))
sensitivity = tp / (tp + fn)
print ("Sensitivity: " + str(sensitivity))
specificity = tn / (tn + fp)
print ("Specificity: " + str(specificity))

df_column_names = ['Date', 'Temperature', 'Humidity', 'Light',
                   'CO2', 'HumidityRatio']
X_column_names = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']

X_test = pd.read_csv(test_data_file, sep='\t', names=df_column_names, usecols=X_column_names)

df_results = pd.read_csv(results_file, sep='\t', names=['y'])
df_results['y'] = df_results['y'].astype('category')

y_true = df_results['y']

y_test_pred = clf_all.predict(X_test)
clf_test_accuracy = accuracy_score(y_true, y_test_pred)
print('Accuracy on test dataset (full model): ' + str(clf_test_accuracy))

try:
    plot_confusion_matrix(clf_all, X_test, y_true)
    plt.show()
except(TypeError):
    plot_confusion_matrix(y_true, y_test_pred, df_results['y'].cat.categories)

try:
    plot_confusion_matrix(clf_all, X_test, y_true, normalize='true')
    plt.show()
except(TypeError):
    plot_confusion_matrix(y_true, y_test_pred, df_results['y'].cat.categories,
                          normalize=True)

df = pd.DataFrame(y_test_pred)
df.to_csv(output_file, index=False, header=False)            

# F measurements task 2

# predictions for CO2
p_co2 = precision_score(y_train_co2,  y_train_pred_co2)
r_co2 = recall_score(y_train_co2,  y_train_pred_co2)
f0_co2= fbeta_score(y_train_co2, y_train_pred_co2, beta=0.5)
f1_co2 = f1_score(y_train_co2,  y_train_pred_co2)
f2_co2 = fbeta_score(y_train_co2,  y_train_pred_co2, beta=2.0)

print ("Precision for CO2 equals: " + str(p_co2))
print ("Recall for CO2 equals: " + str(r_co2))
print ("F0 for CO2 equals: " + str(f0_co2))
print ("F1 for CO2 equals: " + str(f1_co2))
print ("F2 for CO2 equals: " + str(f2_co2))


# predictions for Temperature
p_temp = precision_score(y_train, y_train_pred)
r_temp = recall_score(y_train, y_train_pred)
f0_temp = fbeta_score(y_train, y_train_pred, beta=0.5)
f1_temp = f1_score(y_train, y_train_pred)
f2_temp = fbeta_score(y_train, y_train_pred, beta=2.0)

print ("Precision for Temperature equals: " + str(p_temp))
print ("Recall for Temperature equals: " + str(r_temp))
print ("F0 for Temperature equals: " + str(f0_temp))
print ("F1 for Temperature equals: " + str(f1_temp))
print ("F2 for Temperature equals: " + str(f2_temp))


# predictions for multivariable model
p_all = precision_score(y_train,  y_train_pred_all)
r_all = recall_score(y_train,  y_train_pred_all)
f0_all = fbeta_score(y_train,  y_train_pred_all, beta=0.5)
f1_all = f1_score(y_train,  y_train_pred_all)
f2_all = fbeta_score(y_train,  y_train_pred_all, beta=2.0)

print ("Precision for all variables equals: " + str(p_all))
print ("Recall for all variables equals: " + str(r_all))
print ("F0 for all variables equals: " + str(f0_all))
print ("F1 for all variables equals: " + str(f1_all))
print ("F2 for all variables equals: " + str(f2_all))