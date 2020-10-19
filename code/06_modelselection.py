
# coding: utf-8

# <h1><center>Assignment 06</h1></center>
# <h1><center>Model selection</h1></center>
# <h2><center>Benedek Dankó</h2></center>

# In[2]:


import seaborn as sns
import pandas as pd
from collections import Counter
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib_venn import venn3
from mpl_toolkits.mplot3d import Axes3D

get_ipython().run_line_magic('pylab', 'inline')


# ### 1. Implement a linear model

# In[3]:


# generate a regression model:
X, Y, coef = make_regression(n_samples=1000, n_features=20, random_state=11, coef = True)


# In[4]:


coef # these are the coefficients


# x5, x18, x4 variables have the highest weight, they influence the most the response (Y) variable.

# In[5]:


# get the intercept:
Y[0] - sum(np.multiply(X[0], coef))


# The intercept is actually ~ 0, which means that probably it is subtracted by default (?).

# In[8]:


# Linear regression model:
model = LinearRegression().fit(X, Y)
model.score(X, Y) # perfect fit to the data points


# In[9]:


# coefficients of the Linear Regression model:
model.coef_


# In[10]:


# Difference between the make_regression coefficients and the LinearRegression coefficients:
model.coef_ - coef


# In[11]:


# plot the difference:
fig, ax = plt.subplots(figsize=(8, 6))
plt.plot(model.coef_ - coef)
ax.axes.set_title("make_regression and LinearRegression functions' \ncoefficients difference",fontsize=16)
ax.set_xlabel("Coefficient index",fontsize=13)
ax.set_ylabel("Difference",fontsize=13)
plt.show()


# ### 2. Use of real data

# In[12]:


# load and clean header data:
header = pd.read_csv('../data/communities.names', skiprows = 75, nrows=128, header=None)
header = [i.replace('@attribute ', '').replace(' numeric', '').replace(' string', '') for i in header[0].to_list()]


# In[13]:


df = pd.read_csv('../data/communities.data', names=header, na_values='?')
df.drop(['state', 'county', 'community', 'communityname', 'fold'], axis=1, inplace=True)


# In[14]:


df.head()


# In[15]:


df.describe()


# In[16]:


# Nan values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False)
plt.show()


# In[17]:


# drop columns with too many Nans (threshold: number of rows/2):
df2 = df.dropna(thresh=len(df) - len(df)/2, axis=1)


# In[18]:


plt.figure(figsize=(10, 6))
sns.heatmap(df2.isnull(), cbar=False)
plt.show()


# In[19]:


# fill up remaining NaNs with column means:
df2.fillna(df2.mean(), inplace=True)


# In[20]:


X = np.asarray(df2.loc[:, df2.columns != 'ViolentCrimesPerPop']) # X matrix
Y = np.asarray(df2['ViolentCrimesPerPop']) # Y values


# In[21]:


# KFold cross validation:
kf = KFold(n_splits=5, random_state=12)
clf = LinearRegression() # Linear Regression model


# In[22]:


train_scores = []
test_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    clf.fit(X_train, y_train)
    score_train = clf.score(X_train, y_train)
    score_test = clf.score(X_test, y_test)
    train_scores.append(score_train)
    test_scores.append(score_test)
    print('Train score: {}, test score: {}'.format(score_train, score_test))
    
print('\nMean of training scores: {}, standard deviation of training scores: {}'.format(np.mean(train_scores), np.std(train_scores)))
print('Mean of test scores: {}, standard deviation of test scores: {}'.format(np.mean(test_scores), np.std(test_scores)))


# Of course, the model performs better on the training dataset than on the test dataset (see mean scores), and similarly the SD is higher on the test set.

# In[23]:


lasso = Lasso(random_state=0) # set up model
alphas = np.linspace(0.00001, 1, 100) # generate alpha values
max_iters = [1000, 2000, 5000, 10000] # some max iteration values
norm = [True, False] # normalization True/False

tuned_parameters = [{'alpha': alphas, 'max_iter': max_iters, 'normalize': norm}]
n_folds = 5 # number of folds for validation


# In[24]:


clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, n_jobs=-1, scoring='r2') # GridSearch cross-validation
clf.fit(X, Y) # fit data


# In[25]:


print('Best parameter set: {}'.format(clf.best_params_))


# ### 3. Shrinkage

# In[26]:


# use scaler:
scaler = MinMaxScaler().fit(X)
X_scaled = scaler.transform(X)


# In[27]:


# get alpha, max_iter, normalize parameters:
alphas = clf.cv_results_['param_alpha'].data
max_iters = clf.cv_results_['param_max_iter'].data
norm = clf.cv_results_['param_normalize'].data


# In[28]:


# build model with different alpha parameters (max_iter, normalize don't really influence teh model outcome):

alph = [] # alpha values
all_coeffs = [] # coefficients
r2_scores = [] # R^2 scores

for a in alphas:
    alph.append(a)
    
    l = Lasso(alpha=a, normalize=False)
    l.fit(X_scaled, Y)
    r2_scores.append(l.score(X_scaled, Y))
    all_coeffs.append(l.coef_)


# In[29]:


all_coeffs = np.array(all_coeffs) # convert to array


# In[30]:


plt.plot(alph, all_coeffs)
plt.title('L1 regularization')
plt.ylabel('Coefficients')
plt.xlabel('Penalty parameter')
plt.xlim([0, 0.045])
plt.show()


# In[31]:


plt.plot(alphas, r2_scores)
plt.xlim([0, 0.1])
plt.title('R squared with different penalty parameters')
plt.xlabel('Penalty parameter (alpha)')
plt.ylabel('R squared')
plt.show()


# In[32]:


# with this alpha value we have only 3 coefficients > 0:
list(alphas).index(0.01011090909090909)


# In[33]:


# get those features which are not eliminated with alpha 0.01 (sorry for the ugly code):
pd.Series(list(df2))[[i for i in range(len(list(list(all_coeffs)[8]))) if list(list(all_coeffs)[8])[i] != 0]]


# racePctWhite (percentage of population that is caucasian), PctKids2Par (percentage of kids in family housing with two parents), PctIlleg (percentage of kids born to never married) features still remain with penalty value higher than 0.01.

# In[34]:


# L2 shrinkage:

alph = []
all_coeffs = []
r2_scores = []

for a in np.linspace(0.00001, 10, 50):
    alph.append(a)
    
    r = Ridge(alpha=a, normalize=False)
    r.fit(X_scaled, Y)
    r2_scores.append(r.score(X_scaled, Y))
    all_coeffs.append(r.coef_)


# In[35]:


all_coeffs = np.array(all_coeffs)


# In[36]:


plt.plot(alph, all_coeffs)
plt.title('Ridge regression shrinkage')
plt.xlabel('Penalty parameter (alpha)')
plt.ylabel('Coefficients')
plt.show()


# In[37]:


plt.plot(alph, r2_scores)
plt.title('R squared with different penalty parameters')
plt.xlabel('Penalty parameter (alpha)')
plt.ylabel('R squared')
plt.show()


# In contrast to L1 regularization, L2 regularization did not eliminate (shrinked to 0) the coefficients. L1 regularization results in a sparse coefficient list (many zeros), while L2 regularization just shrinks the coefficients close to zero (because there is a squared term in its equation).

# ### 4. Subset selection

# In[35]:


# Split data:
X_train = X_scaled[::2] # for me it seems, that better to use 
Y_train = Y[::2] #        scaled/normalized data for Ridge, ElasticNet, and maybe for Lasso too
X_test = X_scaled[1::2]
Y_test = Y[1::2]


# In[36]:


# set up 3 different models:
ridge = RidgeCV(alphas=np.linspace(0.0001, 1, 10), cv=5)
lasso = LassoCV(alphas=np.linspace(0.0001, 1, 10), cv=5)
elastic_net = ElasticNetCV(alphas=np.linspace(0.0001, 1, 10), cv=5)


# In[37]:


# set up RFEs, select always 10 features:
selector1 = RFE(ridge, step=1, n_features_to_select=10).fit(X_train, Y_train)
selector2 = RFE(lasso, step=1, n_features_to_select=10).fit(X_train, Y_train)
selector3 = RFE(elastic_net, step=1, n_features_to_select=10).fit(X_train, Y_train)


# In[38]:


len([i for i in list(selector1.support_) if i == True])


# In[39]:


print('Indices of selected features in Ridge regression: {}'.format(np.where(selector1.ranking_ == 1)))
print('Indices of selected features in Lasso regression: {}'.format(np.where(selector2.ranking_ == 1)))
print('Indices of selected features in ElasticNet model: {}'.format(np.where(selector3.ranking_ == 1)))


# As we can see, the 3 different models did not give the same features (as most important ones), but there is some overlap.

# In[40]:


# get the features' actual names:
ridge_names = [list(df2)[i] for i in list(np.where(selector1.ranking_ == 1))[0]]
lasso_names = [list(df2)[i] for i in list(np.where(selector2.ranking_ == 1))[0]]
elasticn_names = [list(df2)[i] for i in list(np.where(selector3.ranking_ == 1))[0]]


# In[41]:


print('Ridge regression most important 10 features: {}\n'.format(ridge_names))
print('Lasso regression most important 10 features: {}\n'.format(lasso_names))
print('Elastic Net most important 10 features: {}\n'.format(elasticn_names))


# Most important features are for example: percentage of population that is african american, percentage of households with wage or salary income in 1989 - Ridge regression; percentage of population that is african american, percentage of males who are divorced - Lasso regression; percentage of population that is african american, percentage of males who are divorced, percentage of females who are divorced - Elastic Net.

# In[42]:


set1 = set(ridge_names)
set2 = set(lasso_names)
set3 = set(elasticn_names)

plt.figure(figsize=(8, 8))
venn3([set1, set2, set3], ('Ridge features', 'Lasso features', 'Elastic net features'))
plt.title('Venn diagram showing the overlap of the most important features of the three models',
         size=17)
plt.show()


# In[43]:


# Create train, test data with only those 10 selected features:
X_train_ridge = X_train[:,list(np.where(selector1.ranking_ == 1)[0])]
X_test_ridge = X_test[:,list(np.where(selector1.ranking_ == 1)[0])]

X_train_lasso = X_train[:,list(np.where(selector2.ranking_ == 1)[0])]
X_test_lasso = X_test[:,list(np.where(selector2.ranking_ == 1)[0])]

X_train_elasticn = X_train[:,list(np.where(selector3.ranking_ == 1)[0])]
X_test_elasticn = X_test[:,list(np.where(selector3.ranking_ == 1)[0])]


# Test the three different models on the test data (only the selected features are included):

# In[44]:


ridge = Ridge(alpha=1, random_state=1)
ridge.fit(X_train_ridge, Y_train)


# In[45]:


ridge.score(X_test_ridge, Y_test)


# In[46]:


lasso = Lasso(random_state=1, alpha=0.00001)
lasso.fit(X_train_ridge, Y_train)


# In[47]:


lasso.score(X_train_ridge, Y_train)


# In[48]:


elastic = ElasticNet(random_state=1, normalize=True, alpha=0.0001)
elastic.fit(X_train_elasticn, Y_train)


# In[49]:


elastic.score(X_test_elasticn, Y_test)


# It seems that for now, the Lasso regression performs the best (however, some hypermarapeter tuning may change the performances).

# In[50]:


elasticn_residuals = Y_test - elastic.predict(X_test_elasticn)
ridge_residuals = Y_test - ridge.predict(X_test_ridge)
lasso_residuals = Y_test - lasso.predict(X_test_lasso)


# In[51]:


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 6))
ax[0].scatter(Y_test, lasso_residuals, color='darkmagenta', s=10, alpha=0.6)
ax[0].axes.set_title('Residuals - L1 regularization',fontsize=16)
ax[0].set_xlabel('Actual y',fontsize=13)
ax[0].set_ylabel('Residuals',fontsize=13)

ax[1].scatter(Y_test, ridge_residuals, color='seagreen', s=10, alpha=0.6)
ax[1].axes.set_title('Residuals - L2 regularization',fontsize=16)
ax[1].set_ylabel('Residuals',fontsize=13)
ax[1].set_xlabel('Actual y',fontsize=13)

ax[2].scatter(Y_test, elasticn_residuals, color='darkorange', s=10, alpha=0.6)
ax[2].axes.set_title('Residuals - L1 + L2 regularization',fontsize=16)
ax[2].set_ylabel('Residuals',fontsize=13)
ax[2].set_xlabel('Actual y',fontsize=13)

fig.tight_layout(pad=6.0)
plt.show()


# From these plots it seems, thath L2 and L1 + L2 regularization performs the best.

# ### 5. ElasticNet penalty surface

# In[52]:


# Calculate MAE (from the example notebook):
def get_mae_for_alpha(alpha, l1_ratio):
    en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    en.fit(X_train, Y_train)
    en_pred = en.predict(X_test)
    return np.absolute((en_pred - Y_test)).mean()


# In[53]:


# Shrinkage with different alpha, L1 ratio parameters:
alphas = []
l1_ratios = []
maes   = []

for a, l in zip(np.linspace(0.0001, 3, 50), np.linspace(0.0001, 0.9999, 50)):
    alphas.append(a)
    l1_ratios.append(l)
    maes.append(get_mae_for_alpha(a, l))


# In[54]:


# this is needed, because the scores are in a 1D array:
def symmetricize(arr1D):
    ID = np.arange(arr1D.size)
    return arr1D[np.abs(ID - ID[:,None])]


# In[55]:


# Plot hyperparameter surface:
fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
X,Y = np.meshgrid(alphas, l1_ratios)
Z = symmetricize(np.asanyarray(maes))
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('Alpha')
ax.set_ylabel('L1 ratio')
ax.set_zlabel('MAE');


# Or using ElasticNetCV for hyperparameter optimization:

# In[56]:


model = ElasticNetCV(l1_ratio=np.linspace(0, 1, 50), alphas=np.linspace(0.0001, 10, 50), cv=5, n_jobs=-1)


# In[57]:


model.fit(X_train, Y_train)


# In[58]:


print('alpha: %f' % model.alpha_)
print('l1_ratio_: %f' % model.l1_ratio_)


# Alpha 0.0001 and L1 ratio 0.816327 hyperparameters minimizes the objective(alpha, beta) in my model. <br>
# I guess for this dataset some non-linear models would perform better.
