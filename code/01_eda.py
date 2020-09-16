#!/usr/bin/env python
# coding: utf-8

# <center><h1> Exploratory data analysis </center>

# <center><h2> Benedek Dankó </center>

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# **1.**

# In[3]:


df = pd.read_csv("../data/titanic.csv") # read csv file


# In[4]:


df.describe() # short summary of the data frame


# In[5]:


df.isna().sum() # check number of missing values per variable


# An elegant Seaborn method to visualize nan values (rows: values, columns: variables):

# In[6]:


plt.figure(figsize=(8, 6))
sns.heatmap(df.isnull(), cbar=False)
plt.show()


# Another method (Matplolib):

# In[45]:


plt.figure(figsize=(25, 200))
plt.imshow(df.isna()) # visualize nan values per variable
plt.show()


# Column 10 with many yellow squares (nan values): Cabin variable. <br>
# Additionally, variable Age has also many nan values (177).

# Replace Cabin, Embarked nan values with 0s:

# In[70]:


df["Cabin"] = df["Cabin"].fillna(0) # replace nan Cabin values with 0s
df["Embarked"] = df["Embarked"].fillna(0) # replace nan Embarked values with 0s


# In[72]:


df.Cabin.head()


# In[74]:


set(df.Embarked)


# In[59]:


len(df["Age"][df["Age"].isna() == True])/len(df["Age"]) # proportion of Age values with NaNs = 19.9%


# Replace Age nan values with the mean of its values:

# In[63]:


df["Age"] = df["Age"].fillna(df["Age"].mean()) # replace nan values in Age column with mean values


# **2.** 

# Count number of persons died/survived in each Pclass:

# In[8]:


df_heatmap = df[['PassengerId', 'Survived', 'Pclass']].groupby(['Pclass', 'Survived']).agg(['count']).reset_index()


# In[9]:


df_heatmap.columns = ["Pclass", "Survived", "Count"] # rename columns


# Format data:

# In[10]:


df_died = df_heatmap[df_heatmap["Survived"] == 0][["Pclass", "Count"]].set_index("Pclass")
df_died.columns = ["Died"]

df_survived = df_heatmap[df_heatmap["Survived"] == 1][["Pclass", "Count"]].set_index("Pclass")
df_survived.columns = ["Survived"]


# Joint table, in the required format:

# In[11]:


df_heatmap = pd.concat([df_died, df_survived], axis=1)
df_heatmap


# Create heatmap:

# In[25]:


ax = sns.heatmap(df_heatmap.to_numpy(), linewidth=0.3)
plt.title("Heatmap of people survived/died in each Pclass category")
plt.xlabel("Died/survived")
plt.ylabel("Pclass")
ax.set(yticklabels=list(df_heatmap.index))
plt.show()


# X axis: 0 or 1 (dead, alive) <br>
# Y axis: Pclass variable (1/2/3) <br>
# Color: number of persons in that category

# **3.**

# Boxplot of Age variable in different Pclass categories:

# In[6]:


f, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x="Pclass", y="Age",
            data=df)
sns.despine(offset=10, trim=True)
ax.axes.set_title("Age distribution in different Pclass categories",fontsize=20)
ax.set_xlabel("Pclass",fontsize=17)
ax.set_ylabel("Age",fontsize=17)
plt.show()


# **4.**

# Correlation matrix of all variables:

# In[5]:


# original code from here: https://www.kaggle.com/timbaney1989/titanic-correlation-map-and-machine-learning

f, ax = plt.subplots(figsize=(10, 6))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cbar=True, cmap="RdYlGn")
ax.axes.set_title("Correltaion matrix of all variables",fontsize=20)
plt.show()


# Interestingly, Pclass variable influenced the most that people survived/died. There's a negative correlation between these variables, however, Survived coded as '0-1', so it is a binary variable. <br>
# After Pclass, Fare variable also influenced survival chances, the higher the fare, the higher the possibility that a giver person survived.

# **5.**

# **a)**

# Scatterplot, describing relationship between 4 variables:

# In[26]:


f, ax = plt.subplots(figsize=(10, 6))
sns.despine(f, left=True, bottom=True)
sns.scatterplot(x="Age", y="Fare",
                hue="Survived",style="Sex",
                sizes=(3, 12), linewidth=0,
                data=df, ax=ax, s=50)
ax.axes.set_title("Scatterplot describing relationship of age, fare, survived, sex variables",fontsize=18)
ax.set_xlabel("Age",fontsize=17)
ax.set_ylabel("Fare",fontsize=17)
plt.show()


# As you may see, most of the people who died (blue) had lower fare. <br>
# Furthermore, most of the little children survived (age 0-5). Over the age of 60, most people could not survive.

# **b)**

# Barplot describing the survival distribution of different age categories:

# In[3]:


# original code/idea from here: https://www.kaggle.com/thulani96/titanic-dataset-analysis-with-seaborn

f, ax = plt.subplots(figsize=(10, 6))
sns.set(font_scale = 1.2)
interval = (0,18,35,60,120)
categories = ['Children','Young adult','Adult', 'Elderly']
df['Age_cats'] = pd.cut(df.Age, interval, labels = categories)

ax = sns.countplot(x = 'Age_cats',  data = df, hue = 'Survived', palette = 'Set2')
ax.axes.set_title("Age Categorical Survival Distribution",fontsize=20)
ax.set_xlabel("Age Categorical",fontsize=17)
ax.set_ylabel("Total",fontsize=17)

plt.show()


# Interestingly, higher fraction of the people in the categories young adult, adult and elderly died, maybe they were more altruist to save other people, like the children. <br>
