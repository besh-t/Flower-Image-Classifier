#PROGRAMMER: BESHER TABBARA
#DATE: 3/31/2019

import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
from image_methods import process_image
from model_methods import load_model
import json

parser = argparse.ArgumentParser(add_help=True)

parser.add_argument('image', type=str, help='path to directory and file name for the image')
parser.add_argument('model', type=str, help='path to directory and file name for the stored model')
parser.add_argument('--top_k', type=int, help='number of top predicted classes to be returned')
parser.add_argument('--category_names', type=str, help='path to directory and file name of categories to labels')
parser.add_argument('--gpu', action="store_true", default=False, help='sets processing device to GPU')

args, _ = parser.parse_known_args()

image = args.image if args.image else ValueError('Path to image required')
model_checkpoint = args.model if args.model else ValueError('Path to model checkpoint required')
top_k = args.top_k if args.top_k else 5
category_names = args.category_names if args.category_names else 'cat_to_name.json'
gpu = args.gpu if args.gpu else False


def predict_image(image_path, model_checkpoint, topk=5, return_classes_labels=False, 
                  category_names=category_names, gpu=False):
    
    #loading checkpoint to processing device
    if gpu and torch.cuda.is_available():
        checkpoint = torch.load(model_checkpoint, map_location='cuda:0')
    else:
        checkpoint = torch.load(model_checkpoint, map_location='cpu')
    
    arch = checkpoint['arch']
    num_labels = len(checkpoint['dir_to_cat'])
    hidden_units = checkpoint['hidden_units']
    model = load_model(arch=arch, num_outputs=num_labels, hidden_units=hidden_units)
    if gpu and torch.cuda.is_available():
        model.cuda()
    
    model.load_state_dict(checkpoint['best_model_weights'])       
    model.eval()
    
    image = process_image(image_path, arch=arch)
    #unsqueezing a single image
    image = torch.tensor(image).float().unsqueeze_(0)
    
    if gpu and torch.cuda.is_available():
        image = image.cuda()

    #no grad calcs
    with torch.no_grad():
        if arch == 'inception' or arch == 'resnet':
            output_ps = model(image)
        else:
            output_ps = torch.exp(model(image))
    
    #get the top k classes and their probs   
    top_p, top_classes = output_ps.topk(topk, dim=1)
    #taking the softmax to get probs (0, 1) for inception and ResNet based models
    if arch == 'inception' or arch == 'resnet':
        top_p = torch.nn.functional.softmax(top_p, dim=1)
    #processing the output tensors to match return format
    probs = list(top_p.detach().cpu().numpy()[0])
    top_dirs = [str(i+1) for i in top_classes.cpu().numpy()[0]]
    #mapping our dirs to the categories
    dirs_dict = checkpoint['dir_to_cat']
    classes = [dirs_dict[dir_key] for dir_key in top_dirs]
    #get actual texts for the classes if return_classes_labels
    if return_classes_labels:
        with open(category_names, 'r') as f:
            labels_dict = json.load(f)
        classes = [labels_dict[class_key] for class_key in classes] #returns the actual label text for classes
    
    if args.image:
        print('Predicted classes and their probabilities:', list(zip(classes, probs)))
        
    return probs, classes


predict_image(image, model_checkpoint, topk=top_k, return_classes_labels=True, 
        category_names=category_names, gpu=gpu)