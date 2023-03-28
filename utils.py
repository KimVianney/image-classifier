from collections import OrderedDict
from os.path import isdir

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms


def load_gpu(gpu):
    if not gpu:
        return torch.device("cpu")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device == "cpu":
        print("WARNING: GPU Not Found! Load CUDA enabled device")
    
    return device


def load_pretrained_model(arch="vgg16"):
    model = models.vgg16(pretrained=True)
    model_name = arch
    
    for param in model.parameters():
        param.no_grad = False
        
    return model



def classifier(model, output_features, hidden_units):
    classifier = nn.Sequential(OrderedDict([
                ('inputs', nn.Linear(25088, hidden_units)), 
                ('relu1', nn.ReLU()),
                ('dropout',nn.Dropout(0.5)), 
                ('hidden_layer1', nn.Linear(hidden_units, 4096)),
                ('relu2',nn.ReLU()),
                ('hidden_layer2',nn.Linear(4096, 2056)),
                ('relu3',nn.ReLU()),
                ('hidden_layer3',nn.Linear(2056, output_features)),
                ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    return model.classifier


def train_model(device, trainloader, testloader, optimizer, model, criterion, epochs=5):
    epochs = epochs
    running_loss = 0
    steps = 0
    print_every = 5


    for epoch in range(epochs):
        for images, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            images, labels = images.to(device), labels.to(device)
        
            #Reset gradient
            optimizer.zero_grad()
        
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()
            
                #Execute validation step
                with torch.no_grad():
                    for images, labels in testloader:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)
                    
                        validation_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {validation_loss/len(testloader):.3f}.. "
                    f"Test accuracy: {accuracy/len(testloader):.3f}")
            
                running_loss = 0
                model.train()

    return model


def validate_model(device, test_loader, model, criterion):
    test_accuracy = 0

    with torch.no_grad():
        model.eval()
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logps = model(images)
            test_loss = criterion(logps, labels)
        
            # Calculate accuracy
            ps = torch.exp(logps)
            top_ps, top_class = ps.topk(1, dim=1) 
            equals = top_class == labels.view(*top_class.shape)
            matches = torch.mean(equals.type(torch.FloatTensor)).item()
            test_accuracy += matches
        
            return f"Test Accuracy for this model: {test_accuracy / len(test_loader) * 100}%"
        
        
def save_checkpoint(model, dir, dataset, output, optimizer):
    if type(dir) == type(None):
        print("Please provide a path to store your model. Your model will not be saved")
        
    else:
        if isdir(dir):
            model.class_to_idx = dataset.class_to_idx

            # Create checkpoint to store model
            checkpoints = {
                'model_arch': 'vgg16_bn',
                'input_size': 25088,
                'output_size': output,
                'classifier': model.classifier,
                'hidden_layers': 8192,
                'class_to_idx': model.class_to_idx,
                'state_dict': model.state_dict(),
                'optimizer_dict': optimizer.state_dict()    
            }

            torch.save(checkpoints, 'new_checkpoint.pth')
            
            
def load_checkpoint(filepath):
    # Load saved checkpoints
    checkpoint = torch.load(filepath)
    
    # Load pretrained model
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.no_grad = False
        
    # Initialize model with saved state_dict
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return model