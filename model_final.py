#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


dataset = pd.read_csv('hiring.csv')

print(dataset.head())


# In[58]:


dataset.columns


# In[60]:


dataset.rename(columns={'test_score(out of 10)':'test_score'})


# In[62]:


dataset.drop(labels = 'test_score(out of 10)',axis=1)


# In[80]:


def convert_to_int(word):
    word_dict = {'one':1,'two':2,'three':3,'four':4, 'five':5, 'six':6, 'seven':7,'eight':8, 
                 'nine':9,'zero':0,'ten':10,'eleven':11, 0:0}
    return word_dict[word]


# In[81]:


dataset['experience'].fillna(0, inplace=True)
dataset['test_score(out of 10)'].fillna(dataset['test_score(out of 10)'].mean(),
                                       inplace = True)


# In[82]:


X = dataset.iloc[:,:3]
X


# In[83]:


X['experience'] = X['experience'].apply(lambda x: convert_to_int(x))
X


# In[84]:


y = dataset.iloc[:,-1]
y


# In[85]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[86]:


#fitting model with training data
regressor.fit(X,y)


# In[89]:


#Saving model to pickle
pickle.dump(regressor, open('model.pkl','wb'))


# In[91]:


#loading model to compare results
model = pickle.load(open('model.pkl','rb'))


# In[93]:


print(model.predict([[2,9,6]]))


# In[ ]:




