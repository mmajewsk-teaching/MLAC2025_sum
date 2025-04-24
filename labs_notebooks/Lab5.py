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

# # MNIST Classification with PyTorch
# In this notebook, we'll extend our knowledge of neural networks to classify handwritten digits from the MNIST dataset using PyTorch.

# %%
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# + [markdown] magic_args="[markdown]"
# ## Part 1: Understanding the MNIST Dataset
#
# MNIST is a widely used dataset in machine learning, consisting of 28x28 grayscale images of handwritten digits (0-9).
# It contains 60,000 training images and 10,000 test images, each with a label indicating which digit it represents.
# -

# ## Read the article about mnist

# [https://en.wikipedia.org/wiki/MNIST_database](https://en.wikipedia.org/wiki/MNIST_database)

# %%
# Define the transformations to apply to our data
transform = transforms.Compose([
    transforms....,                     # Convert images to PyTorch tensors
    transforms.Normalize(...) # Normalize with mean and std of MNIST
])

# Download and load the MNIST dataset
mnist_dataset = ...  # TODO: Load the MNIST dataset with train=True, download=True and apply the transform

# Explore the dataset size
print(f"Dataset size: {len(mnist_dataset)} images")



# %%
# Let's visualize some examples from the dataset
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()
for i in range(10):
    img, label = mnist_dataset[i]
    img = ...  # TODO: Convert the tensor to numpy and remove the channel dimension
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"Digit: {label}")
    axes[i].axis('off')
plt.tight_layout()
plt.show()



# + [markdown] magic_args="[markdown]"
# ## Part 2: Train/Test Split - Why It's Important
#
# Splitting data into training and testing sets is a fundamental practice in machine learning:
#
# 1. **Avoiding Overfitting**: Testing on unseen data helps us evaluate if our model generalizes well
# 2. **Unbiased Evaluation**: Provides an honest assessment of model performance
# 3. **Model Selection**: Helps in selecting the best model architecture/hyperparameters
#
# PyTorch makes this easy with built-in functionality.
# -

# %%
# MNIST actually comes with a predefined test set, but let's create a validation set from our training data
train_size = int(0.8 * len(mnist_dataset))
val_size = len(mnist_dataset) - train_size

# Randomly split the training dataset
train_dataset, val_dataset = ...  # TODO: Split the mnist_dataset into train_dataset and val_dataset with the sizes above

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")

# Create DataLoaders for batch processing
batch_size = 64
train_loader = ...(..., batch_size=batch_size, shuffle=True)
val_loader = ...(..., batch_size=batch_size)

# Get a sample batch
images, labels = next(iter(train_loader))
print(f"Batch shape: {images.shape}")  # Should be [batch_size, 1, 28, 28]
print(f"Labels shape: {labels.shape}")  # Should be [batch_size]

# + [markdown] magic_args="[markdown]"
# ## Part 3: Building a Multi-Layer Perceptron (MLP)
#
# Now we'll build a simple Multi-Layer Perceptron (MLP) to classify the MNIST digits.
#
# Our architecture will be:
# - Input layer: 784 neurons (28x28 pixels flattened)
# - Hidden layer: 128 neurons with ReLU activation
# - Output layer: 10 neurons (one for each digit) with softmax activation
# -

# %%
# Define our MLP model
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        # Network layers
        self.flatten = ...  # TODO: Create a layer to flatten the input images
        self.fc1 = ...      # TODO: Create a linear layer from 28*28 to 128 neurons
        self.relu = ...     # TODO: Define a ReLU activation
        self.fc2 = ...      # TODO: Create a linear layer from 128 to 10 neurons (output)
        
    def forward(self, x):
        # Forward pass through the network
        x = ...  # TODO: Flatten the input
        x = ...  # TODO: Apply the first linear layer
        x = ...  # TODO: Apply ReLU activation
        x = ...  # TODO: Apply the output layer
        return x

# Create an instance of our model and move it to the device (CPU/GPU)
model = MNISTClassifier().to(device)
print(model)

# %%
# Define loss function and optimizer
criterion = ...  # TODO: Define the CrossEntropyLoss
optimizer = ...  # TODO: Define an SGD optimizer with learning rate 0.01 and momentum 0.9

# + [markdown] magic_args="[markdown]"
# ## Part 4: Training the Network
#
# Now we'll train our network using the training data.
# During training, we'll:
# 1. Feed batches of images through the network
# 2. Calculate the loss using cross-entropy
# 3. Backpropagate the gradients
# 4. Update the weights
#
# We'll also periodically evaluate the model on the validation set.
# -

# %%
# Function to calculate accuracy
def calculate_accuracy(model, data_loader, device):
    model....()  # Set model to evaluation mode
    correct = 0
    total = 0
    
    with torch.no_...():  # Disable gradient calculation for inference
        for images, labels in data_loader:
            images, labels = images.to(...), labels.to(...)
            outputs = model(...)
            _, predicted = torch....(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

# %%
# Training loop
num_epochs = 5
train_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the device
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = ...  # TODO: Forward pass through the model
        loss = ...     # TODO: Calculate the loss using criterion, outputs and labels
        
        # Backward pass and optimize
        optimizer....()  # Clear gradients
        loss....()        # Backpropagation
        optimizer....()       # Update weights
        
        running_loss += loss.item()
        
        # Print statistics every 100 mini-batches
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0
    
    # Calculate validation accuracy at the end of each epoch
    val_accuracy = calculate_accuracy(model, val_loader, device)
    val_accuracies.append(val_accuracy)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_accuracy:.4f}')

print('Training finished!')

# %%
# Plot the validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), val_accuracies, marker='o')
plt.title('Validation Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# + [markdown] magic_args="[markdown]"
# ## Part 5: Testing on Unseen Data
#
# Finally, let's evaluate our model on the official MNIST test set, which contains images our model has never seen before.
# This gives us the most accurate assessment of how well our model will perform in real-world scenarios.
# -

# %%
# Load the official MNIST test set
test_dataset = ...  # TODO: Load the MNIST dataset with train=False, download=True and apply the transform
test_loader = ...(..., batch_size=batch_size)

# Calculate accuracy on the test set
test_accuracy = calculate_accuracy(model, test_loader, device)
print(f'Accuracy on the test set: {test_accuracy:.4f}')

# %%
# Visualize some predictions
model.eval()  # Set the model to evaluation mode

# Get the first batch from the test loader
test_images, test_labels = next(iter(test_loader))
test_images, test_labels = test_images.to(device), test_labels.to(device)

# Get predictions
with torch.no_grad():
    outputs = model(test_images)
    _, predicted = torch.max(outputs, 1)

# Move tensors back to CPU for visualization
test_images = test_images.cpu()
test_labels = test_labels.cpu()
predicted = predicted.cpu()

# Display the first 10 images with their true and predicted labels


fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()
for i in range(10):
    img = test_images[i].s...().n...()
    axes[i].imshow(img, cmap='gray')
    color = 'green' if predicted[i] == test_labels[i] else 'red'
    axes[i].set_title(f"True: {test_labels[i]}\nPred: {predicted[i]}", color=color)
    axes[i].axis('off')
plt.tight_layout()
plt.show()


