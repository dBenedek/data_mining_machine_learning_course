#!/usr/bin/env python
# coding: utf-8

# <br></br>
# <h1><center>Assignment 05</h1></center>
# <h1><center>Logistic regression</h1></center>
# <h2><center>Benedek Dankó</h2></center>

# In[177]:


import seaborn as sns
import pandas as pd
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import statsmodels.discrete.discrete_model as sm

get_ipython().run_line_magic('pylab', 'inline')


# #### 1. Download data from https://science.sciencemag.org/content/359/6378/926 (supplementary materials). If you do not succeed, you will find _aar3247_Cohen_SM_Tables-S1-S11.xlsx_ file in the homework's folder.

# In[178]:


# read data, drop unnecessary columns:
df = pd.read_excel('../data/aar3247_Cohen_SM_Tables-S1-S11.xlsx', sheet_name='Table S6', header=2,
                  nrows=1817)

# create list storing whether sample is cancerous (1) or not (0)
df['CancerSEEK Test Result'][df['CancerSEEK Test Result'] == 'Positive'] = 1
df['CancerSEEK Test Result'][df['CancerSEEK Test Result'] == 'Negative'] = 0

cseekY = df['CancerSEEK Test Result'].to_list() # predicted by CancerSEEK
cseek_score = df['CancerSEEK Logistic Regression Score'].to_list() # CancerSEEK model pred. probabilities
trueY = [0 if i != 'Normal' else 1 for i in df['Tumor type'].to_list()] # true Y label, converted to binary

df.drop(['CancerSEEK Test Result', 'CancerSEEK Logistic Regression Score', # drop unnecessary columns
        'AJCC Stage', 'Patient ID #'], axis=1, inplace=True)


# In[179]:


# remove "*" from columns having it, convert it to float type if possible:
for col in list(df):
    try:
        if df[col].str.contains('\*').any() == True:
            print("{} has asterisk".format(col))
            df[col] = df[col].map(lambda x: str(x).lstrip('*').rstrip('*')).astype(float)
    except:
        print("Can't convert {} - string column".format(col))


# In[180]:


# check column types:
g = df.columns.to_series().groupby(df.dtypes).groups
print(g)


# In[181]:


df.describe()


# #### 2. Predict if a sample is cancerous or not

# In[182]:


# Plot nan values:
plt.figure(figsize=(8, 6))
plt.title('NaN values')
sns.heatmap(df.isnull(), cbar=False)
plt.show()


# In[183]:


# fill up NaNs with column means:
df.fillna(df.mean(), inplace=True)


# In[184]:


# split dataset:
trainX = df.drop(['Tumor type', 'Sample ID #'], axis=1).values[::2] # keep only numeric columns (protein levels)
trainY = trueY[::2]
testX  = df.drop(['Tumor type', 'Sample ID #'], axis=1).values[1::2]
testY = trueY[1::2]


# In[185]:


# fit model:
model = LogisticRegression(random_state=11).fit(trainX, trainY)


# In[186]:


# accuracy score:
acc = model.score(testX, testY)
print(acc)


# In[187]:


# preditction: 
predicted = model.predict(testX)
print('First 10 predicted y: {}'.format(predicted[:10]))
print('First 10 y test points: {}'.format(testY[:10]))


# In[188]:


# calculate the fpr and tpr for all thresholds of the classification
# idea from here: https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python

probs = model.predict_proba(testX)
preds = probs[:,1] # probability values for the logistic regression
fpr, tpr, threshold = metrics.roc_curve(testY, preds)
roc_auc = metrics.auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[189]:


# Confusion matrix:
cm = metrics.confusion_matrix(testY, predicted)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Accuracy Score: {0}'.format(acc), size = 13)
plt.show()


# In[190]:


# CancerSEEK ROC curve:
fpr, tpr, threshold = metrics.roc_curve(cseekY[1::2], cseek_score[1::2], pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.title('Receiver Operating Characteristic - CancerSEEK')
plt.plot(fpr, tpr, 'b-', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[191]:


# Confusion matrix:
cm = metrics.confusion_matrix(testY, cseekY[1::2])
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Accuracy Score: {0}'.format(acc), size = 13)
plt.show()


# Their model is great, having AUC 1.0. <br>
# My model's performance is below the CancserSEEK's, with 0.92 AUC. <br> However, the confusion matrix is exactly the same, with the same accuracy. <br><br>
# It seems, that the CancerSEEk method used a different threshold for the logistic regression, which is around 0.9. Over 0.9 probability predictions were classified as Positive (cancer), and below 0.9 as Negative. In my case, the basic threshold is 0.5. This is the reason of having different ROC plots.

# #### 4. Hepatocellular carcinoma

# In[192]:


# filter rows, we need ros with tumor type Liver, or Normal
df_hep = df.loc[(df['Tumor type'] == 'Normal') | (df['Tumor type'] == 'Liver')]


# In[193]:


features = list(df_hep)
features = features[2:27] # first 25 protein level features


# In[194]:


# train X, train Y data:
trainX = df_hep[features]
trainY = [1 if i == 'Liver' else 0 for i in df_hep['Tumor type'].to_list()]


# In[195]:


# Logistic regression model, fit data:
log_reg = sm.Logit(np.asarray(trainY), np.asarray(trainX)).fit()


# In[196]:


# printing the summary table 
print(log_reg.summary()) 


# The 5 best predictor based on P values:
# - HGF (x18, p = 0.000)
# - HE4 (x17, p = 0.001)
# - DKK1 (x10, p = 0.001)
# - AFP (x1, p = 0.006)
# - CD44 (x7, p = 0.008)

# According to [this](https://www.hindawi.com/journals/ijh/2012/859076/) paper, the most common HCC bimoarkers are:
# - AFP
# - GPC3
# - DCP
# - GGT
# - AFU
# - HCR2
# - GOLPH2
# - TGF-Beta 
# - TSGF
# - EGFR family 
# - HGF/SF
# - FGF
# 
# Out of these 12 biomarkers, our model predicted 2 also as a factor associated with HCC.

# #### 5. Multiclass classification

# In[197]:


# training X:
trainX = df.drop(['Tumor type', 'Sample ID #'], axis=1).values[::2]


# In[198]:


tumor_types = list(set(df['Tumor type'].to_list())) # 8 tumor types

Y = [] # converted to numeric (0-8)
for i in df['Tumor type'].to_list():
    Y.append(tumor_types.index(i))


# In[199]:


trainY = Y[::2] # each 2nd value


# In[200]:


# fit model:
model2 = LogisticRegression(multi_class='multinomial', solver ='newton-cg', 
                            random_state=11).fit(trainX, trainY)


# In[201]:


# test data:
testX = df.drop(['Tumor type', 'Sample ID #'], axis=1).values[1::2]
testY = Y[1::2]


# In[202]:


# accuracy score:
acc = model2.score(testX, testY)
print(acc)


# In[203]:


# preditction: 
predicted = model2.predict(testX)
print('First 10 predicted y: {}'.format(predicted[:10]))
print('First 10 y test points: {}'.format(testY[:10]))
print('\n')
print('Last 10 predicted y: {}'.format(predicted[-10:]))
print('Last 10 y test points: {}'.format(testY[-10:]))


# In[204]:


same = 0
for i in range(len(predicted)):
    if predicted[i] == testY[i]:
        same += 1
print('predicted = Y true in {}% of the cases.'.format(same/len(testY)*100))
# btw same as model.score()


# In[205]:


# Confusion matrix:
cm = metrics.confusion_matrix(testY, predicted)
plt.figure(figsize=(8,8))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Accuracy Score: {0}'.format(acc), size = 13)
plt.show()


# It can be confusing that the classes don't have nearly the same number of rows - label 5 has much more rows in total.

# In[206]:


# Idea from here: https://stackoverflow.com/questions/45332410/sklearn-roc-for-multiclass-classification

# plot 8 ROC curves for the predicted classes on the same plot:
def plot_multiclass_roc(clf, X_test, y_test, n_classes, figsize=(17, 6)):
    y_score = clf.decision_function(X_test)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve of the 8 classes', size=15)
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()

plot_multiclass_roc(model2, testX, testY, n_classes=8, figsize=(8, 8))


# In[207]:


tumor_types # where label 0 = Colorectal cancer, label 1 = Liver and so on


# Based on the ROC curves, breast cancer, ovary cancer, and then pancreas, liver cancer can be predicted the most reliably using this multinomial logistic regression model.
