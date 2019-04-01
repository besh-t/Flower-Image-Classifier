#PROGRAMMER: BESHER TABBARA
#DATE: 3/31/2019

import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def process_image(image, arch='vgg16'):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #set the image size, inception requires 299x299
    input_size = 299 if arch == 'inception' else 224
    
    #define transformation
    image_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(input_size),
                                          transforms.ToTensor()])
    
    #open PIL image and apply transformation
    img = image_transform(Image.open(image)).float()
    
    #convert to numpy, normalize and transpose
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (np.transpose(img, (1, 2, 0)) - mean)/std 
    img = np.transpose(img, (2, 0, 1))
    
    return img