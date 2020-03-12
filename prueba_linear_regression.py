import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

headers = ['Population of City in 10,000s','Profit in $10,000s']
df = pd.read_csv('ex1data1.txt', header=None, names=headers)
df.head()

#Particionamos nuestra base de datos
m,n = df.shape #number of training examples
df.insert(0, "0's", np.ones((m,1)), True) #Add a column of ones to x
X = df[["0's", "Population of City in 10,000s"]].values
y = df['Profit in $10,000s'].values.reshape((m,1))

#theta = np.zeros((n,1)) #initialize fitting parameters
iterations = 1500
alpha = 0.01
theta = np.zeros((n,1))

#Cost function
def computeCost(x, y, theta):
    m = len(x)
    hypoth = x @ theta;
    a = hypoth - y;
    return ((a.T @ a)/ (2 * m)).ravel()

#Gradient descent
def gradientDescent(x, y, theta, alpha, num_iters):
    m = len(y)
    J_hist =[]
    
    for i in range(num_iters):
        hypoth = x @ theta
        error = x.T @ (hypoth - y)
        descent = alpha * error * (1/m);
        theta = theta - descent
        J_hist.append(computeCost(x,y,theta))
    
    return theta, J_hist

theta, J_hist = gradientDescent(X,y,theta,0.01,1500)

#Plotting the linear fit
linear_func = lambda x: x @ theta
ax = sns.scatterplot(x=df['Population of City in 10,000s'], 
                     y=df['Profit in $10,000s'], data=df)
plt.plot(df['Population of City in 10,000s'], linear_func(X))