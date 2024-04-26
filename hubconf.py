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
  X,y=make_blobs(n_samples=10,centers=2,n_features=3)
  print(X.shape)
  return X,y
