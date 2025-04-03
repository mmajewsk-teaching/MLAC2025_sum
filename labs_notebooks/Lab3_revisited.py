# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# # Linear Regression Revisited
# This notebook revisits linear regression from first principles using numpy.

import numpy as np
import matplotlib.pyplot as plt

# # Understanding Linear Regression
# We'll explore why we need linear regression and how to implement it from scratch

from sklearn.datasets import fetch_openml
import pandas as pd



# ## Part 1 - using simple equations

# @TODO get the boston dataset
boston = fetch_openml(name=..., version=1, as_frame=True)
df = ....frame

# @TODO plot LSTAT and MEDV variables
plt.figure(figsize=(5, 3))
plt.scatter(df[...], df[...])
plt.xlabel('LSTAT (% lower status of the population)')
plt.ylabel('MEDV (Median value of homes in $1000s)')
plt.title('Boston Housing Dataset: LSTAT vs MEDV')
# Highlight the first data point that we'll focus on
# @TODO pick the first point from the dataset
x_picked, y_picked = ..., ...
plt.scatter(x_picked, y_picked, color="black", s=100)
plt.grid(True)

# Linear regression helps us model the relationship between variables
# We can use it to predict house prices based on neighborhood characteristics
# It's also useful for understanding how variables like LSTAT impact house values

# @TODO we are doing linear regression
def neuron(x, w):
    return ... 

# @TODO add random weights
w = np.array([...,...])

# @TODO assign correct values, use x from dataset
x_1 = ...
x_b = np....(x_1)
x = np....([x_1, x_b]).T
y_pred = neuron(x, w)
error = (df["MEDV"].values - y_pred)
mse = np.mean(error**2)

plt.figure(figsize=(5, 3))
# @TODO plot the boston data
plt.scatter(..., ...)
# @TODO plot predicted line
plt.plot(..., ..., c='r', label='Initial Line')
# @TODO plot picked point in black
plt.scatter(..., ..., color=..., s=100, label='Focus Point')
plt.title(f'MSE: {mse:.2f}')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.legend()
plt.grid(True)

# The line doesn't fit our focus point well yet
# We need a systematic way to find weights that make the line pass through specific points

# Initial random weights

# The correct_weights function calculates weights to make a line pass through a point (x_n, y_n).
#
# For a linear model with equation: y = w_0*x + w_1
# We want to find weights such that: y_n = w_0*x_n + w_1
#
# Using LaTeX notation:
# $y_n = w_0 \cdot x_n + w_1$
#
# To find w_0, we rearrange:
# $w_0 = ...TODO$
#
# This is exactly what our function calculates:
# w_0 = ...TODO
#
# For w_1, we use our new w_0 value:
# $w_1 = ...TODO$
#
# This ensures that our line equation y = w_0*x + w_1 will pass through the point (x_n, y_n).
# Since we have 2 unknowns (w_0 and w_1) but only 1 constraint (the line must pass through one point),
# there are infinitely many solutions. Our approach gives one particular solution.

# @TODO fill in the function
def correct_weights(x_n, y_n, w):
    w_0 = ...
    w_1 = ...
    return w_0, w_1


# Let's verify this works with our focus point
x_n = df['LSTAT'][0]  # x-value of our focus point
y_n = df['MEDV'][0]   # y-value of our focus point

x_n, y_n

# Calculate weights that make our line pass through the focus point
w_0, w_1 = correct_weights(x_n, y_n, w)
exact_weights = np.array([w_0, w_1])

print(f"Focus point: ({x_n}, {y_n})")
print(f"Original weights: {w}")
print(f"Exact weights: {exact_weights}")

# +
# Plot with new weights
plt.figure(figsize=(5, 3))
plt.scatter(df['LSTAT'], df['MEDV'])


# Plot line with exact weights
# @TODO
x_with_bias = np.vstack([..., np.ones_like(...)]).T
# @TODO fill in the weidhts that we have just calculated
y_pred_exact = neuron(x_with_bias, ...)
plt.plot(x_1, y_pred_exact, 'g-', label='Exact Line')
plt.scatter(x_n, y_n, color="black", s=100, label='Focus Point')
plt.title('Line passing through focus point')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.legend()
plt.grid(True)
# -

# ## Part 2 - Applying the equation iteratively

# Starting with our random weights
current_weights = w.copy()
alpha = 0.2  # Learning rate - fraction of correction to apply

# Now lets implement a way to not update the weights straigh away, do it iteratively - this will be useful in next section. In the plot below you should see how the line slowly approaches our "ideal" weights.

# +

# Iteratively adjust weights
fig, axe = plt.subplots(3, 3, figsize=(12, 8))

y_pred_init = neuron(x, current_weights)

for i in range(9):
    # @TODO Calculate the correct weights for picked point, based on current_weights
    w_0_ideal, w_1_ideal = ...(..., ..., ...)
    ideal_weights = np.array([w_0_ideal, w_1_ideal])
    
    # @TODO Calculate the difference between ideal_weights and current weights
    weight_diff = ideal_weights - ...
    # @TODO update current weights based on the fraction of the weight difference
    current_weights = ... + alpha * weight_diff
    
    y_pred = neuron(x, current_weights)
    axe[i//3, i%3].scatter(df['LSTAT'], df['MEDV'], alpha=0.5)
    axe[i//3, i%3].plot(x_1, y_pred_init, 'r--', alpha=0.5, label='Initial Line')
    axe[i//3, i%3].plot(x_1, y_pred, 'g-', label=f'Iteration {i+1}')
    axe[i//3, i%3].set_title(f'Iteration {i+1}')
# -
# # Regression to multiple points

# ## Part 3 - Simple regression to all the points
#
# Now we would like to actually do the regression but to all of the points.
# We will use MSE to be metric for our method.

# Starting with our random weights
current_weights = w.copy()
alpha = 0.02  # Learning rate - fraction of correction to apply

# Iteratively adjust weights
fig, axe = plt.subplots(3, 3, figsize=(12, 8))
current_weights = w.copy()
mse_history = []
for i in range(9):
    for j in range(len(df["LSTAT"])):
        # @TODO each iteration use next point
        x_n = df[...][...]
        y_n = df[...][...]
        # @TODO calculate ideal weights
        w_0_ideal, w_1_ideal = ...(..., ..., ...)
        ideal_weights = np.array([w_0_ideal, w_1_ideal])
        # @TODO Calculate the difference between ideal_weights and current weights
        weight_diff = ideal_weights - ...
        # @TODO update current weights based on the fraction of the weight difference
        current_weights = ... + alpha * weight_diff

        # @TODO calculate prediction
        y_pred = neuron(x, ...)
        error = df["MEDV"].values - y_pred
        mse = np.mean(error**2)
        mse_history.append(mse)
    
    y_pred = neuron(x, current_weights)
    axe[i//3, i%3].scatter(df['LSTAT'], df['MEDV'], alpha=0.5)
    axe[i//3, i%3].plot(x_1, y_pred_init, 'r--', alpha=0.5, label='Initial Line')
    axe[i//3, i%3].plot(x_1, y_pred, 'g-', label=f'Iteration {i+1}')
    axe[i//3, i%3].scatter(x_n, y_n, color="black", s=80, label='Focus Point')
    axe[i//3, i%3].set_title(f'Iteration {i+1}')


# +
plt.figure(figsize=(8, 5))
# @TODO plot mse history
plt.plot(..., 'ro-')
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE over Iterations - Instability in Single-Point Updates')
plt.grid(True)

# The MSE history shows instability as we're updating weights after each point
# This is because each point pulls the weights in a different direction
# The line oscillates between trying to fit different points
# -

# ## Part 4 - Batched regression

# The calculation of each point may be unstable, so we can use batches to stabilise

# +

# Implementing batch gradient descent for more stability
current_weights = w.copy()
alpha = 0.03  # Smaller learning rate for batch updates
epochs = 360
batch_mse_history = []

# Create subplots for batch gradient descent visualization
fig, axe = plt.subplots(3, 3, figsize=(12, 8))

# Reset weights for visualization
current_weights = w.copy()
y_pred_init = neuron(x, current_weights)

# Always include first and last epoch
# For the remaining 7 plots, select epochs based on modulo
plot_count = 0

# Track lines for several epochs
for epoch in range(epochs):
    # Calculate weight adjustments based on all points
    total_adjustment = np.zeros(2)
    for j in range(len(df["LSTAT"])):
        x_n = df['LSTAT'][j]
        y_n = df['MEDV'][j]
        w_0_ideal, w_1_ideal = correct_weights(x_n, y_n, current_weights)
        ideal_weights = np.array([w_0_ideal, w_1_ideal])
        
        # @TODO accumulate weight adjustments
        adjustment = ideal_weights - ...
        total_adjustment += ...
    
    # @TODO pply average adjustment
    avg_adjustment = ... / ...
    current_weights = current_weights + ... * avg_adjustment
    
    # Calculate MSE with new weights
    y_pred = neuron(x, current_weights)
    error = df["MEDV"].values - y_pred
    mse = np.mean(error**2)
    batch_mse_history.append(mse)
    
    # Plot the line at specific epochs to see progression
    # Plot first epoch, last epoch, and 7 epochs in between using modulo
    if epoch == 0 or epoch == epochs-1 or epoch % (epochs // 8) == 0:
        if plot_count < 9:  # Only plot if we have space in our 3x3 grid
            axe[plot_count//3, plot_count%3].scatter(df['LSTAT'], df['MEDV'], alpha=0.5)
            axe[plot_count//3, plot_count%3].plot(x_1, y_pred_init, 'r--', alpha=0.5, label='Initial Line')
            axe[plot_count//3, plot_count%3].plot(x_1, y_pred, 'g-', label=f'Epoch {epoch+1}')
            axe[plot_count//3, plot_count%3].set_title(f'Epoch {epoch+1}')
            axe[plot_count//3, plot_count%3].grid(True)
            plot_count += 1
    

# Add a title to the entire figure of subplots
fig.suptitle('Batch Gradient Descent - Line Evolution', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for the suptitle


# +
# Plot MSE history for batch updates
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), batch_mse_history, 'go-')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE over Epochs - Batch Updates')
plt.grid(True)

# Batch gradient descent produces more stable convergence
# By averaging updates across all points, we get a more balanced direction
# The line gradually fits the overall trend in the data
# -
# We can see that the optimalisation is much more stable but still not good enough.

# ## Part 5 - MSE optimisation
#
# So we have been using mean squared error as a metric, but now we should optimise for it.

# Mean Squared Error (MSE) is defined as the average of squared differences between predictions and actual values
# Given:
# - $X$ is our input matrix with features (LSTAT) and bias term
# - $w$ is our weight vector (slope and intercept)
# - $y$ is our target values (MEDV)
# - $\hat{y} = Xw$ is our prediction
#
# The MSE is calculated as:
# $\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \frac{1}{n}\sum_{i=1}^{n}(y_i - X_i w)^2$
#
# To optimize MSE, we need its gradient with respect to weights $w$:
# $\nabla_w \text{MSE} = \frac{\partial \text{MSE}}{\partial w}$
#
# Expanding the differentiation:
# $\nabla_w \text{MSE} = \frac{\partial}{\partial w}\frac{1}{n}\sum_{i=1}^{n}(y_i - X_i w)^2$
#
# Solving step by step:
# $\nabla_w \text{MSE} = \frac{1}{n}\sum_{i=1}^{n}\frac{\partial}{\partial w}(y_i - X_i w)^2$
#
# Using the chain rule: $\frac{\partial}{\partial w}(y_i - X_i w)^2 = 2(y_i - X_i w)\frac{\partial}{\partial w}(y_i - X_i w)$
#
# Since $\frac{\partial}{\partial w}(y_i - X_i w) = -X_i^T$, we get:
# $\nabla_w \text{MSE} = \frac{1}{n}\sum_{i=1}^{n}2(y_i - X_i w)(-X_i^T) = -\frac{2}{n}\sum_{i=1}^{n}X_i^T(y_i - X_i w)$
#
# In vector form for all samples:
# $\nabla_w \text{MSE} = -\frac{2}{n}X^T(y - Xw) = \frac{2}{n}X^T(Xw - y)$
#
# For gradient descent, we update weights in the opposite direction of the gradient:
# $w_{new} = w_{old} - \alpha \nabla_w \text{MSE}$
#
# Where $\alpha$ is our learning rate. Plugging in the gradient:
# $w_{new} = w_{old} - \alpha \frac{2}{n}X^T(Xw_{old} - y)$
#
# This is our update rule for weights using gradient descent to minimize MSE.
#
# [Linear regression article](https://medium.com/@gurjinderkaur95/linear-regression-and-gradient-descent-in-python-937e77ec68)

# +
# Initialize weights and parameters
current_weights = w.copy()
learning_rate = 0.0001  # Small learning rate for MSE-based updates
epochs = 100
mse_history = []

# Create subplots for MSE-based gradient descent visualization
fig, axs = plt.subplots(3, 3, figsize=(12, 8))

# Initial prediction
y_pred_init = neuron(x, current_weights)

# Convert data to numpy arrays for efficient computation
X = np.vstack([df['LSTAT'], np.ones(len(df['LSTAT']))]).T  # Add bias term
y = df['MEDV'].values

# Track progress for plotting
plot_count = 0

# Run gradient descent with MSE-based updates
for epoch in range(epochs):
    # Compute predictions for all points (just for MSE calculation)
    y_pred = neuron(X, current_weights)
    
    # Compute and record MSE
    error = y - y_pred
    mse = np.mean(error**2)
    mse_history.append(mse)
    
    # Stochastic updates - one point at a time
    for j in range(len(df["LSTAT"])):
        # Select a single data point
        x_j = np.array([df['LSTAT'][j], 1]).reshape(1, 2)  # Single point with bias
        y_j = df['MEDV'][j]  # Single target
        
        # Compute prediction for this point
        y_pred_j = neuron(x_j, current_weights)
        
        # @TODO compute gradient for this single point
        gradient_j = ...
        
        # @TODO update weights immediately after seeing this point
        current_weights = current_weights - ...
    
    # Plot at specific epochs to see progression
    if epoch == 0 or epoch == epochs-1 or epoch % (epochs // 8) == 0:
        if plot_count < 9:  # Only plot if we have space in our 3x3 grid
            row, col = plot_count // 3, plot_count % 3
            axs[row, col].scatter(df['LSTAT'], df['MEDV'], alpha=0.5)
            axs[row, col].plot(df['LSTAT'], y_pred_init, 'r--', alpha=0.5, label='Initial')
            axs[row, col].plot(df['LSTAT'], y_pred, 'b-', label=f'Epoch {epoch+1}')
            axs[row, col].set_title(f'MSE: {mse:.2f} (Epoch {epoch+1})')
            axs[row, col].grid(True)
            axs[row, col].set_xlabel('LSTAT')
            axs[row, col].set_ylabel('MEDV')
            plot_count += 1

# Add a title to the entire figure
fig.suptitle('Gradient Descent with MSE Optimization', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for the suptitle



# +
# Plot MSE history for batch updates
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), mse_history, 'go-')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE over Epochs - Batch Updates')
plt.grid(True)

# Batch gradient descent produces more stable convergence
# By averaging updates across all points, we get a more balanced direction
# The line gradually fits the overall trend in the data

# +
# Final comparison plot after optimization
plt.figure(figsize=(10, 6))
plt.scatter(df['LSTAT'], df['MEDV'], alpha=0.7, label='Data Points')
plt.plot(df['LSTAT'], y_pred, 'b-', linewidth=2, label='Optimized Line (MSE Gradient Descent)')


plt.xlabel('LSTAT (% lower status of the population)', fontsize=12)
plt.ylabel('MEDV (Median value of homes in $1000s)', fontsize=12)
plt.title('Linear Regression: Before and After MSE Optimization', fontsize=14)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
# -




