import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


dataset = pd.read_csv('hiring.csv')

print(dataset.head())


dataset['experience'].fillna(0, inplace=True)
def convert_to_int(word):
    word_dict = {'one':1,'two':2,'three':3,'four':4, 'five':5, 'six':6, 'seven':7,'eight':8, 
                 'nine':9,'zero':0,'ten':10,'eleven':11, 0:0}
    return word_dict[word]


X = dataset.iloc[:,:3]
dataset['experience'].fillna(0, inplace=True)

X['experience'] = X['experience'].apply(lambda x: convert_to_int(x))

y = dataset.iloc[:,:-1]

#splitting training and test set
#training data with all available


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#fitting model with training data
regressor.fit(X,y)


#Saving model to pickle
pickle.dump(regressor, open('model.pkl','rb'))

#loading model to compare results
model = pickle.load(open('model.pkl','rb'))
print(model.print([[2,9,6]]))