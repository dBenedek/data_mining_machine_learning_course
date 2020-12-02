#!/usr/bin/env python
# coding: utf-8

# <h1><center>Assignment 12</h1></center>
# <h2><center>Benedek Dankó</h2></center>
# 

# Data source: https://www.nature.com/articles/s41586-018-0352-3#Sec7

# In[ ]:


# import required packages:

get_ipython().run_line_magic('tensorflow_version', '2.x')
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.metrics import accuracy_score

from google.colab import drive


# #### 1. Load & prepare data

# In[ ]:


# read data:
data = pd.read_csv('../data/Donihue 2018-01-00672 Hurricanes Data.csv', na_values=" ")


# In[ ]:


data.head()


# In[ ]:


def my_encoder(array):
    '''
    Encoding of an array to numeric format.
    '''
    unique_items = list(set(array))
    n = len(unique_items)
    encoded_array = []
    for i in array:
        encoded_array.append(unique_items.index(i))
    return encoded_array


# In[ ]:


cat_cols = ['Hurricane', 'Origin', 'Sex']


# In[ ]:


# convert categorical columns to numeric:
for col in cat_cols:
    data[col] = my_encoder(data[col])


# In[ ]:


data.drop(['ID', 'SumFingers', 'SumToes', 'MaxFingerForce'], axis=1, inplace=True) # drop ID column, and 3 more columns, where only after hurricane measurements exist
data = data.dropna(thresh=data.shape[1]-5).reset_index().drop('index', axis=1) # drop rows with too many Nans (actually, there is only one row...)
data.head() # only numeric values


# In[ ]:


plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False)
plt.title('Missing values', fontsize=20)
plt.show()


# #### 2. T-SNE

# In[ ]:


# select columns (no categorical features):
data_numeric = data.drop(['Sex', 'Hurricane', 'Origin'], axis=1)


# In[ ]:


# scale the data:
data_transformed = MinMaxScaler().fit_transform(data_numeric)


# In[ ]:


tsne = TSNE(random_state = 123)


# In[ ]:


get_ipython().run_cell_magic('time', '', '# measuring time\n# fit model to the the normalized data:\nembedded = tsne.fit_transform(data_transformed)')


# In[ ]:


# plot the data:
fig, ax = plt.subplots(figsize=(8, 6))
ax.axes.set_title('t-SNE colored by Sex',fontsize=16)
for i, j in enumerate(embedded[:,0]):
    if data.Sex.to_list()[i] == 1:
        plt.scatter(embedded[:,0][i] , embedded[:,1][i], c='blue', s=15, marker='o') # male
    else:
        plt.scatter(embedded[:,0][i] , embedded[:,1][i], c='red', s=15, marker='o') # female
plt.axis('off')
plt.show()


# In[ ]:


# plot the data:
fig, ax = plt.subplots(figsize=(8, 6))
ax.axes.set_title('t-SNE colored by Hurricane',fontsize=16)
for i, j in enumerate(embedded[:,0]):
    if data.Hurricane.to_list()[i] == 1:
        plt.scatter(embedded[:,0][i] , embedded[:,1][i], c='black', s=15, marker='o') # after
    else:
        plt.scatter(embedded[:,0][i] , embedded[:,1][i], c='orange', s=15, marker='o') # before
plt.axis('off')
plt.show()


# In[ ]:


# plot the data:
fig, ax = plt.subplots(figsize=(8, 6))
ax.axes.set_title('t-SNE colored by Origin',fontsize=16)
for i, j in enumerate(embedded[:,0]):
    if data.Origin.to_list()[i] == 1:
        plt.scatter(embedded[:,0][i] , embedded[:,1][i], c='purple', s=15, marker='o') # Pine Cay
    else:
        plt.scatter(embedded[:,0][i] , embedded[:,1][i], c='green', s=15, marker='o') # Water Cay
plt.axis('off')
plt.show()


# In[ ]:


# plot the data:
fig, ax = plt.subplots(figsize=(8, 6))
ax.axes.set_title('t-SNE colored by FingerCount',fontsize=16)
for i, j in enumerate(embedded[:,0]):
    if data.FingerCount.to_list()[i] < np.mean(data.FingerCount):
        plt.scatter(embedded[:,0][i] , embedded[:,1][i], c='darkred', s=15, marker='o') # lower finger count (lower than avg.)
    else:
        plt.scatter(embedded[:,0][i] , embedded[:,1][i], c='orange', s=15, marker='o') # higher finger count (>= avg.)
plt.axis('off')
plt.show()


# The best separation is based on the *Sex* variable. <br>
# The other separations are not so nice.

# We got basically two main clusters, which seems to be based on the sex of the lizzards.

# #### 3. Linear model + fine-tune

# In[ ]:


# create x matrix, y array:
y = data.Hurricane
x = data.drop('Hurricane', axis=1).to_numpy()


# GridSearch cross-validation, with 5-fold cross-validation, using Logisitc Regression with L2 regularization:

# In[ ]:


params = {'C': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 5, 10]}

# default cv: 5-fold cross-validation
gs = GridSearchCV(estimator=LogisticRegression(random_state=1001, penalty='l2', max_iter=1000), 
                      param_grid=params, 
                      verbose=1,  # verbose: the higher, the more messages
                      scoring='accuracy', 
                      return_train_score=True)

gs.fit(x, y)


# In[ ]:


print(f'Best C parameter found: {gs.best_params_["C"]}')


# In[ ]:


lr = LogisticRegression(random_state=1001, penalty='l2', max_iter=500, C=gs.best_params_['C']).fit(x, y)


# In[ ]:


probs = lr.predict_proba(x)
preds = probs[:,1] # probability values 
fpr, tpr, threshold = metrics.roc_curve(y, preds)
roc_auc = metrics.auc(fpr, tpr)


# In[ ]:


# Plot the data:
plt.figure(figsize=(6,6))
plt.title('Receiver Operating Characteristic', fontsize=16)
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate', fontsize=14)
plt.xlabel('False Positive Rate', fontsize=14)
plt.show()


# #### 4. SVM + fine-tune

# Firstly, let's scale the data (the x variables):

# In[ ]:


x_scaled = scale(data.drop('Hurricane', axis=1))
y_scaled = data.Hurricane.to_numpy()


# Run a KFold cross-validation with linear kernel:

# In[ ]:


params = {'C': [0.1, 1, 10, 100, 500], 
          'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
          'kernel': ['rbf', 'poly']}

# default cv: 5-fold cross-validation
gs = GridSearchCV(estimator=SVC(), 
                      param_grid=params, 
                      verbose=1,  # verbose: the higher, the more messages
                      scoring='accuracy', 
                      return_train_score=True)

gs.fit(x_scaled, y_scaled)


# In[ ]:


print(f'Best parameter settings: {gs.best_params_}')


# Best model found:

# In[ ]:


svm = SVC(C=gs.best_params_['C'], 
          gamma=gs.best_params_['gamma'], 
          kernel=gs.best_params_['kernel'], 
          random_state=11).fit(x, y)


# In[ ]:


probs = lr.predict_proba(x)
preds = probs[:,1] # probability values 
fpr, tpr, threshold = metrics.roc_curve(y, preds)
roc_auc = metrics.auc(fpr, tpr)


# In[ ]:


# Plot the data:
plt.figure(figsize=(6,6))
plt.title('Receiver Operating Characteristic', fontsize=16)
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate', fontsize=14)
plt.xlabel('False Positive Rate', fontsize=14)
plt.show()


# #### 5. RF + feature importances

# In[ ]:


params = {'max_depth' : [4, 6, 8, 10, 15, 20, 30],
          'n_estimators' : [50, 100, 200, 300]}


# In[ ]:


# default cv: 5-fold cross-validation
gs = GridSearchCV(estimator=RandomForestClassifier(), 
                  param_grid=params, 
                  verbose=1,  # verbose: the higher, the more messages
                  scoring='accuracy', 
                  return_train_score=True,
                  n_jobs=-1)

gs.fit(x, y)


# In[ ]:


print(f'Best parameter settings: {gs.best_params_}')


# Random Forest Classifier with the best parameter settings:

# In[ ]:


rf = RandomForestClassifier(random_state=11, 
                            max_depth=gs.best_params_['max_depth'], 
                            n_estimators=gs.best_params_['n_estimators']).fit(x, y)


# In[ ]:


probs = lr.predict_proba(x)
preds = probs[:,1] # probability values 
fpr, tpr, threshold = metrics.roc_curve(y, preds)
roc_auc = metrics.auc(fpr, tpr)


# In[ ]:


# Plot the data:
plt.figure(figsize=(6,6))
plt.title('Receiver Operating Characteristic', fontsize=16)
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate', fontsize=14)
plt.xlabel('False Positive Rate', fontsize=14)
plt.show()


# In[ ]:


# store the 5 most important features:
top_features = [x for _,x in sorted(zip(rf.feature_importances_, data.columns), reverse=True)][:5]


# In[ ]:


# create dataframe:
df = pd.DataFrame({'Feature': top_features, 
                     'Importance': sorted(rf.feature_importances_, reverse=True)[:5]})


# In[ ]:


# plot the data:
sns.set(rc={'figure.figsize':(9.7,6.27)})
ax = sns.barplot(x='Feature', y='Importance', data=df)
plt.title('The five most important features', fontsize=17)
plt.xticks(rotation=70)
plt.show()


# Based on this, the most important feature is the *SVL* variable.
