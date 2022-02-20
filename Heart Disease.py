#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Prediction 
# 
# ### 

# #### To predict whether a person has heart disease or not

# Let us start by importing the necessary libraries

# In[1]:


import numpy as np    # used for making numpy arrays   
import pandas as pd    # used for creating structured panda dataframes
from sklearn.model_selection import train_test_split    # for splitting data into training and test data
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score    # for evaluation


# #### Importing the data

# In[2]:


heart_data = pd.read_csv(r'C:\Users\Shrudin\Desktop\Hub\Coding\project\Heart Disease\archive\heart.csv')

# Heart Disease UCI dataset from kaggle

# pd.read_csv is used for loading the csv data into Pandas DataFrame


# In[3]:


heart_data.shape # to find the number of rows and columns


# In[4]:


# printing the first 5 rows of the dataset 
heart_data.head()


# In[5]:


# printing the last 5 rows of the dataset 
heart_data.tail()


# In[6]:


table = pd.DataFrame (heart_data) 
table


# In[8]:


# for checking if there are any missing entries in the dataset
heart_data.isnull().sum()


# In[9]:


# we can see that there are no missing entries


# Now let us see the statistical overview of the entire data
heart_data.describe().round(decimals=2)


# In[10]:


# to find the distribution of people affected by heart disease
heart_data['target'].value_counts()


# ### Here,
# 
# #### 0 represents that the person is not affected by heart disease
# #### 1 represents that the person is affected by heart disease

# Since the 'target' column is what needed to be predicted, we have to split the target column from rest of the features.

# ### Splitting the features and target

# In[11]:


Y = heart_data['target']
X = heart_data.drop(columns='target',axis=1)


# In[12]:


X


# In[13]:


Y


# ### Splitting the data into Training data and Test data

# Let us use the train_test_split function that we have imported

# In[14]:


xtrain, xtest, ytrain, ytest= train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=3)

# X is the features
# Y is the target
# test size 0.2 represents that 20% of the data is test data
# stratify is used so that both training and test data contains equal proportions of target, in this case 0,1
# random state for randomizing the output


# In[15]:


print(X.shape, xtrain.shape, xtest.shape)

# to see the rows and columns of X, xtrain, xtest


# ### Model Training

# We are using LogisticRegression model as this is a binary classification problem

# In[16]:


model =  LogisticRegression()


# #### Training the LogisticRegressio model using training data 

# In[17]:


model.fit( xtrain, ytrain )


# ### Model Evaluation

# In[18]:


# Let us find the accuracy on the training data

xtrain_prediction = model.predict(xtrain)
training_data_accuracy = accuracy_score(xtrain_prediction, ytrain)

print ("Accuracy on training data is ", training_data_accuracy)


# In[19]:


# Let us find the accuracy on the test data

xtest_prediction = model.predict(xtest)
test_data_accuracy = accuracy_score(xtest_prediction, ytest)

print ("Accuracy on the test data is ", test_data_accuracy)


# ### Building a predictive system

# In[21]:


input_data = (48,1,0,124,274,0,0,166,0,0.5,1,0,3)


# In[22]:


# to change the input data from tuple into numpy array for easier reshaping
nparray = np.asarray(input_data)

# reshaping the numpy array as we want the prediction for only one instance
reshaped_array = nparray.reshape(1,-1)
prediction = model.predict(reshaped_array)


# Let us print the final prediction

# In[23]:


print (prediction)

if (prediction==1):
    print("The person has heart disease.")
else: 
    print("The person does not have any heart disease.")

