import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.utils import shuffle

url="http://bit.ly/w-data"
df=pd.read_csv(url)

###### GRAPH #######
df.plot(x="Hours", y="Scores", style='o')
plt.title("Hours of studying vs Scores of the student")
plt.xlabel("Hours of studying")
plt.ylabel("Scores of the student")
plt.show()

"""Training the datat"""
x=np.array(df.drop(['Scores'],1))
y=np.array(df['Scores'])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
linear=LinearRegression()
linear.fit(x_train,y_train)
y_predict=linear.predict(x_test)
print(x_test)
print(y_predict)
