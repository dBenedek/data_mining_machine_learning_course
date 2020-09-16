#!/usr/bin/env python
# coding: utf-8

# The woldbank_development_2015.csv file contains the World Development Indicators for the 2015 year, downloaded from The World Bank's [webpage](https://databank.worldbank.org/source/world-development-indicators).

# <br></br>
# <br></br>
# <h1><center>Unsupervised learning & clustering</h1></center>

# <h2><center>Benedek Dankó</h2></center>

# In[212]:


import seaborn as sns
import pandas as pd
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

get_ipython().run_line_magic('pylab', 'inline')


# **1. Reading data**

# In[213]:


# read in table, set nan values:
df = pd.read_csv('woldbank_development_2015.csv', na_values= ['..', 'Nan'])


# In[214]:


df.tail(8)
# looks like there are some empty/"fake" rows at the end:


# In[215]:


# keep only "normal" rows:
df = df.loc[0:379367]


# In[216]:


df.tail(8)
# now it looks OK


# In[217]:


df.isna().sum() # check number of missing values per variable
# 172946 measured features (2015 [YR2015] ) have nans


# In[218]:


# check country/region names, number of names
Counter(df["Country Name"])


# In[219]:


df.loc[0:311829].tail()
# last row index with country, and not region: 311828


# In[220]:


# pivote data frame (keep only countries)
# one row - one country, one column - one feature:
pivoted = df.loc[0:311828].pivot(index='Country Name', columns='Series Code', values='2015 [YR2015]')


# In[221]:


pivoted.head()


# In[222]:


# format numbers (I don't prefer scientific notation in case of smaller numbers)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


# In[223]:


pivoted.head()


# **2. Data preprocessing and inspection**

# Visualizing missing values (Seaborn/Matplotlib):

# In[224]:


sns.set(font_scale=1)
plt.figure(figsize=(12, 8))
sns.heatmap(pivoted.isnull(), cbar=False)
plt.title('Missing values (white)', fontsize=20)
plt.show()
# not all variable/country is visible
# but the pattern is clear


# In[225]:


plt.figure(figsize=(25, 200))
plt.imshow(pivoted.isna().T) # visualize nan values per variable
plt.show()
# rows: variables
# columns: countries
# white: nan, black: normal values


# Calculate missing values:

# In[226]:


# dataframe containing each feature with the number of missing values in it:
features_missing = pivoted.isna().sum().to_frame()


# In[227]:


# dataframe containing each country/region (row) with the number of missing values in it:
countries_missing = pivoted.T.isna().sum().to_frame()


# In[228]:


# select features with less than 20 nans:
selected_features = features_missing[features_missing[0] < 20].index.to_list()


# In[229]:


# select rows with less than 700 nans:
selected_countries = countries_missing[countries_missing[0] < 700].index.to_list()


# In[230]:


# select final dataframe based on the previously filtered rows/columns
df_filtered = pivoted[[c for c in pivoted.columns if c in selected_features]] # filter columns
df_filtered = df_filtered[df_filtered.index.isin(selected_countries)] # filter rows


# In[231]:


df_filtered.tail()


# Visualize nans after filtering data:

# In[232]:


plt.figure(figsize=(12, 8))
sns.heatmap(df_filtered.isnull(), cbar=False)
plt.title('Missing values after filtering', fontsize=20)
plt.show()
# not all variable/country is visible
# much less white bands, less nans


# In[233]:


# fill nans with the mean values of that given column:
df_filtered = df_filtered.fillna(df_filtered.mean())


# In[234]:


df_filtered.shape
# 116 countries, 163 features


# In[235]:


# series name - code pairs:
df.iloc[:, 2:4].head()


# Based on this:
# Hungary seems similar to Slovakia or Romania, Norway seems similar to Germany or Denmark.

# **3. PCA**

# Create model, fit data:

# In[236]:


get_ipython().run_cell_magic('time', '', "n = 3\npca = PCA(n_components=n, random_state=10)\npca_transformed = pca.fit_transform(df_filtered)\nprint('{}% of the variance is explained by the first {} components\\n'.format(round(pca.explained_variance_ratio_.cumsum()[2]*100, 5), n))")


# In[237]:


# create dataframe for plotting:
df_pca = pd.DataFrame(list(zip(pca_transformed[:,0], pca_transformed[:,1], pca_transformed[:,2])), 
               columns =['pca0', 'pca1', 'pca2']) 


# In[238]:


# create pairplots of the 3 components:
g = sns.pairplot(df_pca)
g.fig.set_size_inches(7,7)
g.fig.suptitle("PCA 2D comparisons", y=1.05)
plt.show()


# It does not look good, because of the different scales (eg. 0 - 2.5 Vs. 0 - 1 * 1e16).

# Create model, fit data after normalization, using sklearn MinMaxScaler for normalization:

# In[239]:


transformer = MinMaxScaler().fit(df_filtered) # setting up normalizer
tr = transformer.transform(df_filtered) # transform data
pca_transformed_norm = pca.fit_transform(tr)

df_pca_norm = pd.DataFrame(list(zip(pca_transformed_norm[:,0], 
                                    pca_transformed_norm[:,1], 
                                    pca_transformed_norm[:,2])), 
               columns =['pca0', 'pca1', 'pca2'])


# In[240]:


# create pairplots of the 3 normalized components:
g = sns.pairplot(df_pca_norm)
g.fig.set_size_inches(7,7)
g.fig.suptitle("PCA 2D comparisons after normalization", y=1.05)
plt.show()


# Now it looks much better, points are separated clearly.

# Just to check a few rows of the PCA-transformed data (with the countries labelled):

# In[241]:


fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(pca_transformed_norm[:,0][0:20], pca_transformed_norm[:,1][0:20])
ax.axes.set_title("PCA with the first 21 countries",fontsize=16)
ax.set_xlabel("PCA0",fontsize=13)
ax.set_ylabel("PCA1",fontsize=13)

for i, txt in enumerate(df_filtered.index.to_list()[0:20]):
    ax.annotate(txt, (pca_transformed_norm[:,0][i], pca_transformed_norm[:,1][i]))
plt.show()


# PCA with all countries (Hungary: red, Norway: orange):

# In[242]:


fig, ax = plt.subplots(figsize=(8, 6))
ax.axes.set_title("PCA with the all of the countries",fontsize=16)
ax.set_xlabel("PCA0",fontsize=13)
ax.set_ylabel("PCA1",fontsize=13)
for i, j in enumerate(pca_transformed_norm[:,0]):
    if df_filtered.index.to_list()[i] == 'Hungary':
        plt.scatter(pca_transformed_norm[:,0][i] , pca_transformed_norm[:,1][i], color='red', alpha=1, marker='x', s=75)
    elif df_filtered.index.to_list()[i] == 'Norway':
        plt.scatter(pca_transformed_norm[:,0][i] , pca_transformed_norm[:,1][i], color='orange', alpha=1, marker='x', s=75)
    else:
        plt.scatter(pca_transformed_norm[:,0][i] , pca_transformed_norm[:,1][i], color='blue', alpha=0.5)
plt.show()


# **4. t-SNE**

# In[243]:


# importance of random states
# set up TSNE
tsne = TSNE(random_state = 11)


# In[244]:


get_ipython().run_cell_magic('time', '', '# measuring time\n# fit model to the the normalized data:\nembedded = tsne.fit_transform(tr)')


# In[245]:


# embedded data has 2 components, 163 points (countries)
shape(embedded)


# In[246]:


# plot the data:
fig, ax = plt.subplots(figsize=(10, 8))
ax.axes.set_title("t-SNE",fontsize=16)
for i, j in enumerate(embedded[:,0]):
    if df_filtered.index.to_list()[i] == 'Hungary':
        plt.scatter(embedded[:,0][i] , embedded[:,1][i], c='red', s=150, marker='x')
    elif df_filtered.index.to_list()[i] == 'Norway':
        plt.scatter(embedded[:,0][i] , embedded[:,1][i], c='orange', s=150, marker='x')
    else:
        plt.scatter(embedded[:,0][i] , embedded[:,1][i], c='black', s=10)

for i, txt in enumerate(df_filtered.index.to_list()):
    if txt == 'Hungary' or txt == 'Norway':
        ax.annotate(txt, (embedded[:,0][i], embedded[:,1][i]), color='red', size=20)
    else:
        ax.annotate(txt, (embedded[:,0][i], embedded[:,1][i]), color='black', size=9)
plt.axis('off')
plt.show()


# The closest ones to Norway: Sweden, Switzerland, Finland, Denmark
# <br> and to Hungary: Bulgaria, Slovakia, Poland, Romania

# **5. Hierarchical clustering**

# From the Seaborn manual: d = Rectangular data for clustering. **Cannot contain NAs.**

# In[247]:


df_clean = df_filtered.fillna(df_filtered.mean()) # fill nans with column mean
tr_2 = transformer.transform(df_clean) # normalize data, otherwise would be difficult to visualize


# In[248]:


# create dataframe for clustering, from the normalized data:
df_clust = pd.DataFrame(data=tr_2[0:,0:],    # values
                        index=df_clean.index.to_list(),    
                        columns=tr_2[0,0:]) 
df_clust.columns = list(df_clean) # set column names (features)


# In[249]:


# Plot the data:
sns.set(font_scale=0.7)
h=sns.clustermap(df_clust, cmap="Blues", 
                 linewidth=.5, method="average", 
                 annot=False, figsize=(12, 20), 
                 yticklabels=1, xticklabels=1)
plt.show()


# Here, Hungary clusters with Czech Republic, Croatia, Poland, and Slovakia, similarly to the previosuly shown methods. <br>
# In this case, the clusters of the features are also indicated (at the top) which can be useful too. <br>
# Rows with similar patterns: similar countries (described by these features), columns with similar patterns: similar features. <br>
# Also, interesting to see, how China and the USA is separated from the other clusters (see at the bottom).
