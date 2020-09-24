
# coding: utf-8

# # Supervised learning introduction, K-Nearest Neighbors (KNN)

# <br></br>
# <h1><center>Assignment 03</h1></center>
# <h2><center>Benedek Dankó</h2></center>

# In[1]:


import seaborn as sns
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

get_ipython().run_line_magic('pylab', 'inline')


# **1. Read data**

# In[2]:


# column names:
column_names = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']


# In[3]:


# read data:
data = pd.read_csv('glass.data', delimiter=',', 
                   names=column_names)


# In[4]:


data.head()


# In[5]:


# drop Id column:
data = data.drop('Id', axis=1)
data.head()


# In[6]:


# create corrected Type column (values 1-6):
type_list = [num-1 if num > 4 else num for num in data.Type.to_list()]
data['Type'] = type_list


# In[7]:


# number of different Type values (= 6)
n_types = len(set(type_list))


# In[8]:


def create_converted_type(value, n_types):
    '''
    Returns one-hot encoded list of value.
    Eg. value 3 with 3 classes --> [0,0,1]
    '''
    result_list = [0 for n in range(n_types)]
    result_list[value-1] = 1
    return result_list


# In[9]:


# converted Type column:
Type_converted = [create_converted_type(x, n_types) for x in type_list]


# In[10]:


# add it to the data dataframe:
data['Type_converted'] = Type_converted


# In[11]:


data.head()


# **2. & 3. Implement KNN**

# Two helper functions:

# In[12]:


def l2_distance(point1, point2):
    '''
    Calculates the L2 distance between two data points.
    '''
    point1 = np.array(point1) 
    point2 = np.array(point2)
    return np.linalg.norm(point1 - point2)


# In[13]:


def get_neighbors(train_x, test_x, k):
    '''
    Returns the k nearest neighbors of points in test_x
    in the format of: (train_x point, test_x point, distance)
    '''
    total_distances = list() # holds all distances for each train_X point
    for test_point in test_x:
        distances = list() # holds distances from each point for a given train_x point
        for train_point in train_x:
            dist = l2_distance(test_point, train_point)
            distances.append((train_point, test_point, dist)) # train_X, train_Y, distance
        distances.sort(key=lambda tup: tup[2]) # sort by descending distance
        total_distances.append(distances)
    neighbors = list()
    for l in range(len(total_distances)):
        local_neighbors = list()
        for i in range(k): # we only need the k nearest neighbors
            local_neighbors.append(total_distances[l][i])
        neighbors.append(local_neighbors)
    return neighbors


# The final KNN Classifier function:

# In[14]:


def knn_classifier(x_train, x_test, y_train, k):
    '''
    Returns the number of points in each class divided by k.
    '''
    unique_classes = [] # each type of unique classes, in order (one-hot coded)
    for i in range(len(y_train[0])): 
        unique_classes.append([]) 
        for j in range(len(y_train[0])): 
            if i == j:
                unique_classes[i].append(1)
            else:
                unique_classes[i].append(0)     
    neighbors = get_neighbors(x_train, x_test, k) # get k nearest neighbors of each point
    classes = [] # holds the k nearest neighbors' class (neighbors of the predicted points)
    for point in neighbors:
        point_k_neighbors_class = []
        for neigh in point:
            point_k_neighbors_class.append(y_train[x_train.index(neigh[0])])
        classes.append(point_k_neighbors_class)
    predictions = [] # predicted probabilities for each y_train point
    for point in classes:
        point_probs = []
        for c in unique_classes:
            if list(c) in point:
                prob = point.count(list(c))/k # number of k neighbors with class x/k
            else:
                prob = 0.0 # no k neighbor with that class
            point_probs.append(prob)
        predictions.append(point_probs)
    return predictions


# In[15]:


k = 2
X_train = [[0.9, 0.2, 0.8] , [-1.2, 1.5, 0.7], [5.8, 0.0, 0.9], [6.2, 0.9, 0.9]]
y_train = [[0, 1], [0, 1], [1, 0], [0, 1]]
X_test  = [[0.8, 0.8, 0.6], [0.5, 0.4, 0.3]]

# KNN classification test:
knn_classifier(X_train, X_test, y_train, k)
# format: {new_point_A: class_1: 1.0, new_point_B: class_1: 0.5, class_B: 0.5} - where values are probabilities (between 0-1)


# This means, that for the first predicted y_train point there is 100% that its class is \[0, 1] and the same for the second point too.

# **4. Predictions & interpretation**

# Create training, test datasets in the required format:

# In[16]:


X_train = data.iloc[::2, :] # each 2nd row
Y_train = data.iloc[::2, :]['Type_converted'].to_list() # the known labels
X_train = X_train.drop(['Type', 'Type_converted'], axis=1) # we just separated it to Y_train
X_train.reset_index(drop=True, inplace=True)
X_train_list =[] # re-format (one row - one list)
for row in X_train.itertuples(): 
    tmp_list =[row.RI, row.Na, row.Mg, row.Al, row.Si, row.K, row.Ca, row.Ba, row.Fe] 
    X_train_list.append(tmp_list) 
  
X_test = data.iloc[1:] # the same, but not with the same rows
X_test = X_test.iloc[::2, :]
Y_test = X_test['Type'].to_list()
X_test = X_test.drop(['Type', 'Type_converted'], axis=1)
X_test.reset_index(drop=True, inplace=True)

X_test_list =[] # store in the required format
for rows in X_test.itertuples(): 
    tmp_list =[rows.RI, rows.Na, rows.Mg, rows.Al, rows.Si, rows.K, rows.Ca, rows.Ba, rows.Fe] 
    X_test_list.append(tmp_list) 


# In[17]:


# prediction:
class_pred = knn_classifier(X_train_list, X_test_list, Y_train, 5)


# In[18]:


# pediction probabilities (0-1) for each class:
class_pred[1:10]


# We can see for each predicted point (Y_train points) the probability of belonging to each class.

# In[19]:


# Convert it to simple format:
Type_predicted = []
for point in class_pred:
    Type_predicted.append(point.index(max(point))+1)
    
Type_predicted[:10]


# In[20]:


# Create confusion matrix
confusion_matrix(Y_test, Type_predicted)


# Here, the "x" axis is the predicted label, and the "y" axis is the true label. Higher values in the left to right diagonal line (where predicted = true label) indicate more accurate model.

# In[21]:


# Calculate accuracy:
accuracy_score(Y_test, Type_predicted)


# **5. Compare it to Sklearn's KNN**

# In[22]:


data.head()


# In[23]:


# Y_train in simple format:
Y_train_sckit = data.iloc[::2, :]['Type'].to_list()


# In[29]:


# Create, fit model:
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_list, Y_train_sckit)


# In[25]:


#Predict output
predicted = model.predict(X_test_list) 
print(predicted) 
# predicted class for each point:


# In[26]:


# Check accuracy score:
accuracy_score(Y_test, predicted)


# We can see, that the scikit-learn model has exactly the same results, same accuracy, ehich is 70%.
