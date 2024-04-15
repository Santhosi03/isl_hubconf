import torch
def print_new():
  print("print_new")
  return "hubconf"

import torchvision.models as models

model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')


def get_model():
  print("model")
  return model
