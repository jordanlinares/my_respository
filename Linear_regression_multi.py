#Análisis de datos
import pandas as pd
import numpy as np

headers = ['Size','Number of bedrooms', 'Price']
df_2 = pd.read_csv('ex1data2.txt', header=None, names=headers)

#Particionamos nuestra base de datos
m,n = df_2.shape 
X = df_2[['Size', 'Number of bedrooms']]
y = df_2['Price'].values.reshape((m,1))

mean = X.mean()
std = X.std()
X_nom = (X - mean) / std
X_nom.insert(0, "1's", np.ones((m,1)), True)
X = X_nom.values

#Run gradient descent
#Choose some alpha value
alpha = 0.1;
num_iters = 400;

#Init Theta and Run Gradient Descent 
theta = np.zeros((3, 1));

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

theta, J_history = gradientDescent(X, y, theta, alpha, num_iters);
p = (np.array([[1, 1.650, 3]]) @ theta).ravel()
price = p * 10000
print("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): {:.2f}".format(price[0]))

def normalEqn(x,y):
    I = np.linalg.inv(x.T @ x)
    return I @ (x.T @ y)

theta_2 = normalEqn(X, y)