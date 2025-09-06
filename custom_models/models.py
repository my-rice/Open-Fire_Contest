import torch
from firenetV2 import FireNetV2
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights

def build_MobileNetV3Small(num_outputs=1):
  model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_small', pretrained=True)
  model.classifier = torch.nn.Sequential(
    torch.nn.Linear(in_features=576,
                    out_features=1024, # same number of output units as our number of classes
                    bias=True),
    torch.nn.Hardswish(inplace=True),
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1024,
                    out_features=num_outputs, # same number of output units as our number of classes
                    bias=True))
  return model

def build_MobileNetV2(num_outputs=1):
  model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
  model.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.2, inplace=False),
                                         torch.nn.Linear(in_features=1280, out_features=num_outputs, bias=True))
  return model

def build_FireNetV2():
  model = FireNetV2()
  return model

def build_ResNet50(num_outputs=1):
  model = resnet50(ResNet50_Weights.DEFAULT) # .DEFAULT = best available weights
  model.fc=torch.nn.Linear(in_features=2048,
                    out_features=num_outputs, # same number of output units as our number of classes
                    bias=True)
  return model

def build_ResNet18(num_outputs=1):
  model = resnet18(ResNet18_Weights.DEFAULT) # .DEFAULT = best available weights
  model.fc=torch.nn.Linear(in_features=512,
                    out_features=num_outputs, # same number of output units as our number of classes
                    bias=True)
  return model

def network_parameters_ResNet18(model):
  for param in model.parameters():
    param.requires_grad = False
  model.fc.requires_grad_(True)
  return model

def network_parameters_ResNet50(model):
  for param in model.parameters():
    param.requires_grad = False
  model.fc.requires_grad_(True)
  return model

def network_parameters_MobileNetV2(model):
  for param in model.parameters():
    param.requires_grad = False
  for param in model.classifier.parameters():
    param.requires_grad = True
  return model

def network_parameters_MobileNetV3Small(model):
  for param in model.parameters():
    param.requires_grad = False
  for param in model.classifier.parameters():
    param.requires_grad = True
  return model

def optimizer_settings_ResNet50(model, lr, weight_decay, momentum):
  return torch.optim.SGD(model.fc.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

def optimizer_settings_ResNet18(model, lr, weight_decay, momentum):
  return torch.optim.SGD(model.fc.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

def optimizer_settings_MobileNetV2(model, lr, weight_decay, momentum):
  return torch.optim.SGD(model.classifier.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

def optimizer_settings_MobileNetV3Small(model, lr, weight_decay, momentum):
  return torch.optim.SGD(model.classifier.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
