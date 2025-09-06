import torch
from torch import nn
from PIL import Image
from torchvision import transforms

from keras.models import load_model
import numpy as np
import torchinfo

class FireNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.Conv2D1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3) # Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=X.shape[1:]))
        self.activation1 = nn.ReLU() 
        self.pool1 = nn.AvgPool2d(kernel_size = 2)  # model.add(AveragePooling2D())
        self.dropout1 = nn.Dropout(p=0.5) # model.add(Dropout(0.5))

        self.Conv2D2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3) # model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
        self.activation2 = nn.ReLU() 
        self.pool2 = nn.AvgPool2d(kernel_size = 2) # model.add(AveragePooling2D())
        self.dropout2 = nn.Dropout(p=0.5) # model.add(Dropout(0.5))

        self.Conv2D3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3) # model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        self.activation3 = nn.ReLU()
        self.pool3 = nn.AvgPool2d(kernel_size = 2) # model.add(AveragePooling2D())
        self.dropout3 = nn.Dropout(p=0.5) # model.add(Dropout(0.5))
        
        self.flatten = nn.Flatten() # model.add(Flatten())
        
        self.dense4 = nn.Linear(64*6*6, 256) # model.add(Dense(units=256, activation='relu'))
        self.activation4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.2) # model.add(Dropout(0.2))

        self.dense5 = nn.Linear(256, 128) # model.add(Dense(units=128, activation='relu'))
        self.activation5 = nn.ReLU()

        self.dense6 = nn.Linear(128, 2) # model.add(Dense(units=2, activation = 'softmax'))
        self.activation6 = nn.Softmax(dim=-1)

    def forward(self, x, verbose=False):
        # forward function executed when an input is passed to the nn
        if verbose:
            print("Input shape", x.shape)
        x = self.Conv2D1(x)
        if verbose:
            print("Input shape", x.shape)
        x = self.activation1(x)
        x = self.pool1(x)
        if verbose:
            print("Input shape", x.shape)
        x = self.dropout1(x)
        if verbose:
            print("Input shape", x.shape)

        
        x = self.Conv2D2(x)
        if verbose:
            print("Input shape", x.shape)
        x = self.activation2(x)
        x = self.pool2(x)
        if verbose:
            print("Input shape", x.shape)
        x = self.dropout2(x)
        if verbose:
            print("Input shape", x.shape)

        x = self.Conv2D3(x)
        if verbose:
            print("Input shape", x.shape)
        x = self.activation3(x)
        x = self.pool3(x)
        if verbose:
            print("Input shape", x.shape)
        x = self.dropout3(x)
        if verbose:
            print("Input shape", x.shape)

        x = self.flatten(x)
        if verbose:
            print("Input shape", x.shape)

        
        x = self.dense4(x)
        if verbose:
            print("Input shape", x.shape)
        x = self.activation4(x)
        x = self.dropout4(x)
        if verbose:
            print("Input shape", x.shape)

        x = self.dense5(x)
        if verbose:
            print("Input shape", x.shape)
        x = self.activation5(x)
        

        x = self.dense6(x)
        if verbose:
            print("Input shape", x.shape)
        x = self.activation6(x)
        

        return x
    
    @staticmethod
    def compute_output(output):
        # function to compute the output of the network
        return output[0] # the output is Fire
        # if output[0] > output[1]: # the output is Fire
        #     return output[0]
        # else: # the output is No Fire
        #     return output[1]



def build_FireNet():
    trained_model = load_model('/FireNet_v1_PRETRAINED_weights/Fire-64x64-color-v7-soft.h5')
    weights=trained_model.get_weights()

    model = FireNet()
    

    model.Conv2D1.weight.data = torch.from_numpy(np.transpose(weights[0]))
    model.Conv2D1.bias.data = torch.from_numpy(weights[1])
    model.Conv2D2.weight.data = torch.from_numpy(np.transpose(weights[2]))
    model.Conv2D2.bias.data = torch.from_numpy(weights[3])
    model.Conv2D3.weight.data = torch.from_numpy(np.transpose(weights[4]))
    model.Conv2D3.bias.data = torch.from_numpy(weights[5])
    model.dense4.weight.data = torch.from_numpy(np.transpose(weights[6]))
    model.dense4.bias.data = torch.from_numpy(weights[7])
    model.dense5.weight.data = torch.from_numpy(np.transpose(weights[8]))
    model.dense5.bias.data = torch.from_numpy(weights[9])
    model.dense6.weight.data = torch.from_numpy(np.transpose(weights[10]))
    model.dense6.bias.data = torch.from_numpy(weights[11])
    return model    
