# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import pandas as pd
boston = fetch_openml(name="boston", version=1, as_frame=True)
df = boston.frame

x_min, x_max = df["LSTAT"].min(), df["LSTAT"].max()
x_1 = np.linspace(x_min, x_max, 100)
x_2 = np.zeros_like(x_1) + 1
X = np.vstack((x_1, x_2)).T

def neuron2(x,w):
    return x@w

  # Negative slope, intercept 30
Y2 = neuron2(X, w2)
# -

plt.figure(figsize=(5, 3))
plt.scatter(df['LSTAT'], df['MEDV'])
plt.plot(X[:, 0], Y2, c='red')


# Define our mean squared error loss function
def mse_loss(X, y, w):
    y_pred = neuron2(X, w)
    squared_errors = (y_pred - y) ** 2
    return np.mean(squared_errors)


# Define a function to calculate the gradient of MSE loss
def mse_gradient(X, y, w):
    y_pred = neuron2(X, w)
    errors = y_pred - y
    gradient = (2/len(y)) * X.T @ errors
    return gradient


X_true

# +
y_true = df['MEDV']
X_true = np.vstack((df['LSTAT'], np.zeros_like(df["LSTAT"].values))).T

#mse_loss(X_true, y_true, w )
mse_gradient(X_true, y_true,w )
# -

X_train = np.column_stack([df['LSTAT'], np.ones(len(df))])
y_train = df['MEDV'].values


def batch_gradient_descent(X, y, learning_rate=0.01, n_iterations=100, w=None):
    loss_history = []
    w_history = []
    for i in range(n_iterations):
        loss = mse_loss(X, y, w)
        loss_history.append(loss)
        w_history.append(w.copy())
        
        grad = mse_gradient(X, y, w)
        
        w = w - learning_rate * grad
        
    return w, loss_history, w_history


# Define a function for batch gradient descent
def batch_gradient_descent(X, y, learning_rate=0.01, n_iterations=100, initial_w=None):

    if initial_w is None:
        w = np.zeros(X.shape[1])
    else:
        w = initial_w.copy()
    
    # Initialize history lists
    loss_history = []
    w_history = []
    
    # Perform gradient descent
    for i in range(n_iterations):
        # Calculate loss
        loss = mse_loss(X, y, w)
        loss_history.append(loss)
        w_history.append(w.copy())
        
        grad = mse_gradient(X, y, w)
        
        w = w - learning_rate * grad
        
    return w, loss_history, w_history


w = np.array([4.61, 1])

w, lh, wh = batch_gradient_descent(X_train, y_train,n_iterations=100, w=w)

plt.plot(lh)

#plt.plot(lh)
#plt.plot(wh)
Y2 = neuron2(X, w2)
plt.figure(figsize=(5, 3))
plt.scatter(df['LSTAT'], df['MEDV'])
plt.plot(X[:, 0], Y2, c='red')


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


# +

def tanh(x):
    return np.tanh(x)


# -

def tanh_derivative(x):
    return 1 - np.tanh(x)**2


x = np.linspace(-10,10,100)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))



Y2 = neuron3(X, w2)
plt.figure(figsize=(5, 3))


plt.plot(x, sigmoid(x))
plt.plot(X[:, 0], Y2, c='red')

# %%
# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Visualize the Iris dataset (using petal dimensions)
plt.figure(figsize=(4, 3))
colors = ['red', 'green', 'blue']
markers = ['o', 's', '^']
feature_indices = [2, 3]  # Using petal length and width
for i, c, m in zip(range(3), colors, markers):
    plt.scatter(
        X[y == i, feature_indices[0]], 
        X[y == i, feature_indices[1]], 
        c=c, marker=m, s=60,
        edgecolors='k', label=f"{iris.target_names[i]}"
    )
plt.xlabel(iris.feature_names[feature_indices[0]])
plt.ylabel(iris.feature_names[feature_indices[1]])
plt.title("Iris Dataset - Petal Dimensions")
plt.legend()
plt.grid(True)

X[y == 0, feature_indices[0]], X[y == 0, feature_indices[1]], 


def neuron3(x,w):
    return sigmoid(x@w)
#def neuron3(x,w):
#    return x@w


w = np.array([4.61, 2, 1])
w = np.array([4.61, 2.3, -11])
X1 = np.vstack((X[y == 0, feature_indices[0]], X[y == 0, feature_indices[1]], np.zeros_like(X[y == 0, feature_indices[0]])+1)).T
X2 = np.vstack((X[y == 1, feature_indices[0]], X[y == 1, feature_indices[1]], np.zeros_like(X[y == 1, feature_indices[0]])+1)).T
X_new = np.vstack((X1,X2))
Y_proba = neuron3(X_new,w)

Y_proba

Y_new = (Y_proba>0.5)*1

a = np.zeros_like(X[y == 1, feature_indices[0]])
b = a+1

Y_true = np.hstack((a,b))



Y_true == Y_new


