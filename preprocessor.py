"""
This module defines the main functions and classes to process the data needed by 
model to properly perform classification tasks

"""

from PIL import Image
import numpy as np

import torch
from torchvision import datasets, models, transforms


def train_transformer(train_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_dataset



def test_transformer(test_dir):
    """
    Creates a test dataset suitable for evaluating the model
    """
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_dataset
    

def data_loader(data, train=True):
    
    """
    Initializes data loader for training dataset and test dataset
    Parameters:
        data: Dataset to be loaded
        train: Set to True or False, to determine whether to load training dataset 
                or test dataset
                True (default) - loads training dataset
                False - loads test dataset
    """
    
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=32)
    return loader