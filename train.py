#PROGRAMMER: BESHER TABBARA
#DATE: 3/31/2019

import torch
from torchvision import datasets, transforms
import argparse
from model_methods import train_model

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('data_dir', type=str, help='path to image dataset directory')
parser.add_argument('--save_dir', type=str, help='path and directory to where checkpoint to be saved')
parser.add_argument('--arch', type=str, help='underlying architecture for the neural network model (VGG16, AlexNet, DenseNet 161, inception v3, ResNet 18)')
parser.add_argument('--hidden_units', type=int, help='number of hidden units, only applies to VGG16, AlexNet, DenseNet')
parser.add_argument('--learning_rate', type=float, help='learning rate of the model optimizer')
parser.add_argument('--epochs', type=int, help='number of epochs to train the model')
parser.add_argument('--gpu', action="store_true", default=False, help='sets processing device to GPU')


args, _ = parser.parse_known_args()

save_dir = args.save_dir if args.save_dir else 'checkpoint.pth'
arch = args.arch if args.arch else 'vgg16'
hidden_units = args.hidden_units if args.hidden_units else 4096
learning_rate = args.learning_rate if args.learning_rate else 0.001
epochs = args.epochs if args.epochs else 20
gpu = args.gpu if args.gpu else False

#inception requires image size (299x299)
input_size = 299 if arch == 'inception' else 224

#prepare training and validation image data sets
if args.data_dir:
    image_transforms = {'train' : transforms.Compose([transforms.Resize(256),
                                                      transforms.RandomResizedCrop(input_size),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                                           [0.229, 0.224, 0.225])]),
                        'valid' : transforms.Compose([transforms.Resize(256),
                                                      transforms.CenterCrop(input_size),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                                           [0.229, 0.224, 0.225])]),
                        'test' : transforms.Compose([transforms.Resize(256),
                                                     transforms.CenterCrop(input_size),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                                          [0.229, 0.224, 0.225])])}
    
    image_datasets = {x : datasets.ImageFolder(args.data_dir+'/'+x, transform=image_transforms[x])
                      for x in list(image_transforms.keys())}
    
    image_dataloaders = {x : torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True)
                         for x in list(image_datasets.keys())}
    
    num_class_labels = len(image_datasets['train'].classes)
    
train_model(image_dataloaders, arch=arch, num_outputs=num_class_labels, hidden_units=hidden_units, 
                learning_rate=learning_rate, epochs=epochs, gpu=gpu, checkpoint=save_dir)

