import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
df = pd.read_csv("audi.csv")
# print(df.head())

# print(df.corr())

# print(df.info())
x = df[['year','mpg','engineSize']]
y = df.price
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
print(y_train)
model = LinearRegression()
model.fit(x_train,y_train)
print(model.predict(x_test))
print(model.score(x_test,y_test))
pickle.dump(model,open('model.pkl','wb'))
