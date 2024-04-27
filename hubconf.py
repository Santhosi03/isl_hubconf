import torch
def print_new():
  print("print_new")
  return "hubconf"

import torchvision.models as models
from sklearn.datasets import make_blobs

model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')


def get_model():
  print("model")
  return model

def get_data_blobs(n_points=100):
  X,y=make_blobs(n_samples=n_points,centers=2,n_features=3)
  print(X.shape)
  return X,y

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

def get_data(type, split=None):
    """
    Retrieves the specified dataset and splits it into training and testing sets.

    Args:
        type (str): Name of the dataset to retrieve.
        split (str, optional): Split ratio for train-test split (e.g., "70-30"). Defaults to None.

    Returns:
        tuple: A tuple containing the training data and testing data.
    """
    # Check the dataset type and retrieve the corresponding dataset
    if type.lower() == "scikit-minist-digits":
        data = load_digits()
        X = data.data
        y = data.target
    # Add more dataset options as needed
    
    # Split the data if split ratio is provided
    if split:
        split_ratio = [int(x) for x in split.split("-")]
        print(split_ratio)
        train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=split_ratio[1]/100, random_state=42)
        return (train_data, test_data, train_labels, test_labels)
    else:
        return X, y

