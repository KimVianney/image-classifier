import json
import argparse
from PIL import Image
import numpy as np
import torch
import preprocessor
from torchvision import models, datasets, transforms
from utils import load_gpu, load_checkpoint



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image',type=str, action="store", help='Path to image for prediction')
    parser.add_argument('checkpoint',type=str, help='Model checkpoint')
    parser.add_argument('--topk', type=int, help='Specifies top K matches as ')
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")
    
    args = parser.parse_args()
    
    return args


def process_image(image):
    image = Image.open(image).convert('RGB')
    image.thumbnail((256, 256))
    width, height = image.size
    
    # Crop the image
    n_width, n_height = 224, 224
    crop_params = np.array([
        (width - n_width),
        (height - n_height),
        (width + n_width),
        (height + n_height)]) / 2
    
    # Crop image to be fed into network
    image = image.crop((crop_params[0], crop_params[1], crop_params[2], crop_params[3]))
    
    # Transform image to tensor
    tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], 
                                     [0.229, 0.224, 0.225])
    
    image_tensor = normalize(tensor(image))
    return image_tensor


def predict(image, model, device, cat_to_name, topk=5):
    if not topk:
        topk = 5
    else:
        topk = topk
    #Get processed image
    processed_image = process_image(image)
    processed_image = torch.unsqueeze(processed_image,0).to(device).float()
    
    #Prepare model for eval
    model.to(device)
    model.eval()
    
    #Begin evaluation
    logps = model(processed_image)
    ps = torch.exp(logps)
    top_ps, top_idx = ps.topk(topk, dim=1)
    
    top_ps = np.array(top_ps.detach())[0]
    top_idx = np.array(top_idx.detach())[0]
    
    # Convert to classes
    # idx_to_class = {value: key for key, value in skeleton['class_to_idx'].items()}
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    
    top_labels = [idx_to_class[idx] for idx in top_idx]
    flower_names = [cat_to_name[idx] for idx in top_labels]
    
    return top_ps, top_labels, flower_names

def print_predictions(probs, flowers):
    for idx, preds in enumerate(zip(probs, flowers)):
        print(f"Rank: {idx+1}")
        print(f"Flower: {preds[0]}, Likelihood(0 - Not likely, 1 - Likely): {preds[1]}")


def main():
    args = parse_args()
    
    with open(args.category_names, 'r') as f:
        	cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)
    
    # image_tensor = process_image(args.image)
    
    device = load_gpu(gpu=args.gpu);
    
    top_probs, top_labels, top_flowers = predict(args.image, model, device, cat_to_name,args.topk)
    
    print_predictions(top_flowers, top_probs)

if __name__ == '__main__':
    main()