import argparse
import json

import torch
from torch import nn
from torch import optim

import utils # Contains functions for training models
from utils import validate_model, save_checkpoint
import preprocessor # Contains functions for processing images


# Commandline parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', action="store")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=8192)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=5)
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    args = parser.parse_args()
    return args


with open('cat_to_name.json', 'r') as file:
    cat_to_name = json.load(file)
    
output_classes = len(cat_to_name)

def main():
    # Get Keyword Args for Training
    args = parse_args()
    
    # Set directory for training
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Pprepare the datasets required 
    train_dataset = preprocessor.train_transformer(train_dir)
    valid_dataset = preprocessor.test_transformer(valid_dir)
    test_dataset = preprocessor.test_transformer(test_dir)
    
    trainloader = preprocessor.data_loader(train_dataset)
    validloader = preprocessor.data_loader(valid_dataset, train=False)
    testloader = preprocessor.data_loader(test_dataset, train=False)
    
    # Initialize learning rate
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning rate specificed as 0.001")
    else:
        learning_rate = args.learning_rate
    
    # Create model required for training
    model = utils.load_pretrained_model(arch=args.arch)
    
    
    # Define classifier for model
    model.classifier = utils.classifier(model, output_classes, hidden_units=args.hidden_units)
    
    # Transfer model to gpu 
    device = utils.load_gpu(gpu=args.gpu);
    model.to(device)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Train and store the model after training
    trained_model = utils.train_model(device, trainloader, validloader, optimizer, model, criterion, args.epochs)
    
    print("\nTraining complete!!")
    
    # Validate trained model
    validation_result = validate_model(device, testloader, trained_model, criterion)
    print(validation_result)
   
    # Save checkpoint for model
    save_checkpoint(trained_model, args.save_dir, train_dataset, output_classes, optimizer)
    
    
    
if __name__ == '__main__':
    main()